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
from core.arena_zones import (
    CENTER,
    PERIPHERY,
    center_polygon,
    center_threshold_for_area_fraction,
    classify_center_periphery,
    max_wall_distance,
    valid_step_distances,
    wall_distances,
)
from core.zone_events import (
    UNDETECTED,
    debounce_zone_labels,
    dwell_frames,
    extract_zone_transitions,
    overlapping_zones_to_candidates,
)

logger = logging.getLogger(__name__)

UNCALIBRATED_IMMOBILITY_UNIT = "N/A (uncalibrated)"
NOT_AVAILABLE = "N/A"

# How the open-field center is defined (Freestyle.arena_center_mode).
CENTER_BY_DISTANCE = "distance_cm"
CENTER_BY_AREA_FRACTION = "area_fraction"
CENTER_MODES = (CENTER_BY_DISTANCE, CENTER_BY_AREA_FRACTION)

DEFAULT_CENTER_DISTANCE_CM = 10.0
DEFAULT_CENTER_AREA_FRACTION = 0.25

# Batch/summary columns added only when the open-field arena mode is on.
ARENA_METRIC_HEADERS = [
    'center_time_s', 'periphery_time_s', 'center_entries',
    'latency_to_center_s', 'thigmotaxis_time_pct',
    'thigmotaxis_distance_pct', 'mean_wall_distance',
    'center_speed_mean', 'periphery_speed_mean',
]


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

    # Open-field arena results. Underscore-prefixed so the summary JSON's
    # generic "specific_metrics" dump skips them: arena runs report them under
    # freestyle_metrics['arena'], and runs without an arena stay unchanged.
    _arena_metrics: Optional[Dict[str, any]] = None
    _arena_labels: List[str] = field(default_factory=list)
    _arena_wall_distances: List[Optional[float]] = field(default_factory=list)

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
        if self._arena_metrics is not None:
            base_dict['freestyle_metrics']['arena'] = dict(self._arena_metrics)

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
        # Open-field arena mode (center/periphery, thigmotaxis). Off by
        # default; when off nothing about the outputs changes.
        self.arena_mode = False
        self.arena_center_mode = CENTER_BY_DISTANCE
        self.arena_center_distance_cm = DEFAULT_CENTER_DISTANCE_CM
        self.arena_center_area_fraction = DEFAULT_CENTER_AREA_FRACTION

    @property
    def arena_enabled(self) -> bool:
        """True when an arena polygon is requested and arena metrics are on."""
        return self.arena_mode

    def configure_arena(
        self,
        enabled: bool,
        center_mode: Optional[str] = None,
        distance_cm: Optional[float] = None,
        area_fraction: Optional[float] = None,
    ) -> None:
        """Turn the open-field arena mode on/off and set the center rule."""
        mode = self.arena_center_mode if center_mode is None else center_mode
        if mode not in CENTER_MODES:
            raise ValueError(
                f"center mode must be one of {', '.join(CENTER_MODES)}"
            )
        distance = (
            self.arena_center_distance_cm if distance_cm is None
            else float(distance_cm)
        )
        fraction = (
            self.arena_center_area_fraction if area_fraction is None
            else float(area_fraction)
        )
        if not distance > 0 or distance == float('inf'):
            raise ValueError("center distance must be a positive number of cm")
        if not 0.0 < fraction < 1.0:
            raise ValueError("center area fraction must be between 0 and 1")
        self.arena_mode = bool(enabled)
        self.arena_center_mode = mode
        self.arena_center_distance_cm = distance
        self.arena_center_area_fraction = fraction

    def center_threshold(
        self, arena_points, scale_factor: float
    ) -> Tuple[float, str]:
        """Return (center wall-distance threshold in px, rule actually used).

        A distance rule needs calibration (``scale_factor`` is cm per pixel);
        without it the area-fraction rule is used instead.
        """
        if self.arena_center_mode == CENTER_BY_DISTANCE and scale_factor > 0:
            return (
                self.arena_center_distance_cm / scale_factor,
                CENTER_BY_DISTANCE,
            )
        return (
            center_threshold_for_area_fraction(
                arena_points, self.arena_center_area_fraction
            ),
            CENTER_BY_AREA_FRACTION,
        )

    def center_outline(self, arena_points, scale_factor: float):
        """Center-region polygon (Nx2 px) for overlays, or None if empty."""
        threshold, _ = self.center_threshold(arena_points, scale_factor)
        return center_polygon(arena_points, threshold)

    def update_roi_overlay(self, roi_manager, scale_factor: float) -> None:
        """Cache the center outline on ``roi_manager`` for ``draw_on_frame``.

        Does nothing (and clears the overlay) without an arena or arena mode.
        """
        arena = getattr(roi_manager, 'arena', None)
        if not self.arena_mode or arena is None:
            roi_manager.set_center_overlay(None)
            return
        roi_manager.set_center_overlay(
            self.center_outline(arena.points, scale_factor)
        )

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
        
        if self.arena_mode:
            self._calculate_arena_metrics(result, calibrated, speed_scale)

        logger.info("Freestyle metrics calculation complete")

    @staticmethod
    def _mean_or_none(values) -> Optional[float]:
        values = list(values)
        return float(np.mean(values)) if values else None

    def _calculate_arena_metrics(
        self,
        result: FreestyleAnalysisResult,
        calibrated: bool,
        unit_scale: float,
    ) -> None:
        """Center/periphery occupancy, thigmotaxis and wall-distance metrics.

        Frames are classified against the arena polygon, then debounced with
        the same minimum dwell as user zones. Times and entries come from the
        debounced labels; ``unit_scale`` converts pixels to cm when calibrated.
        """
        result._arena_metrics = {name: None for name in ARENA_METRIC_HEADERS}
        result._arena_labels = []
        result._arena_wall_distances = []
        arena = result.arena_polygon
        if arena is None:
            result.notes.append(
                "Open-field arena metrics are not available: no arena "
                "polygon was defined for this video."
            )
            return

        threshold_px, basis = self.center_threshold(
            arena, result.scale_factor if calibrated else 0.0
        )
        if (
            self.arena_center_mode == CENTER_BY_DISTANCE
            and basis != CENTER_BY_DISTANCE
        ):
            result.notes.append(
                "Center defined by area fraction "
                f"({self.arena_center_area_fraction:g} of the arena): the "
                "distance rule needs spatial calibration. Draw a reference "
                "line to calibrate."
            )
        if threshold_px >= max_wall_distance(arena):
            result.notes.append(
                "The center region is empty: the center threshold is at "
                "least the arena's inscribed radius, so all in-arena time "
                "is periphery."
            )

        positions = result.positions
        candidates = classify_center_periphery(positions, arena, threshold_px)
        min_dwell = (
            dwell_frames(self.min_dwell_seconds, result.fps)
            if result.fps > 0 else 1
        )
        labels = debounce_zone_labels(candidates, min_dwell)
        distances_px = wall_distances(positions, arena)
        frame_time = 1.0 / result.fps if result.fps > 0 else 0.0

        in_arena = [label in (CENTER, PERIPHERY) for label in labels]
        center_time = labels.count(CENTER) * frame_time
        periphery_time = labels.count(PERIPHERY) * frame_time
        tracked_time = center_time + periphery_time

        transitions = extract_zone_transitions(labels)
        center_entries = sum(1 for _, to, _ in transitions if to == CENTER)
        latency = None
        for index, label in enumerate(labels):
            if label == CENTER:
                origin = result.timestamps[0] if result.timestamps else 0.0
                latency = result.timestamps[index] - origin
                break

        # Same valid-step rule as Analyzer.finalize_result; a step belongs to
        # the zone of the frame it ends in.
        valid = [
            pos is not None and not (
                index < len(result.roi_labels)
                and result.roi_labels[index] == UNDETECTED
            )
            for index, pos in enumerate(positions)
        ]
        steps = valid_step_distances(positions, valid, result.jump_limit_px)
        center_distance = sum(
            step for step, label in zip(steps, labels)
            if step is not None and label == CENTER
        )
        periphery_distance = sum(
            step for step, label in zip(steps, labels)
            if step is not None and label == PERIPHERY
        )
        arena_distance = center_distance + periphery_distance

        wall_inside = [
            max(0.0, distance) * unit_scale
            for distance, flag in zip(distances_px, in_arena)
            if flag and distance is not None
        ]
        speeds = result.smoothed_speeds
        center_speeds = [
            speeds[i] for i, label in enumerate(labels)
            if label == CENTER and i < len(speeds) and speeds[i] is not None
        ]
        periphery_speeds = [
            speeds[i] for i, label in enumerate(labels)
            if label == PERIPHERY and i < len(speeds) and speeds[i] is not None
        ]

        result._arena_labels = labels
        result._arena_wall_distances = [
            None if distance is None else distance * unit_scale
            for distance in distances_px
        ]
        result._arena_metrics = {
            'center_time_s': center_time,
            'periphery_time_s': periphery_time,
            'center_entries': center_entries,
            'latency_to_center_s': latency,
            'thigmotaxis_time_pct': (
                100.0 * periphery_time / tracked_time if tracked_time > 0
                else None
            ),
            'thigmotaxis_distance_pct': (
                100.0 * periphery_distance / arena_distance
                if arena_distance > 0 else None
            ),
            'mean_wall_distance': self._mean_or_none(wall_inside),
            'center_speed_mean': self._mean_or_none(center_speeds),
            'periphery_speed_mean': self._mean_or_none(periphery_speeds),
            'center_definition': basis,
            'center_threshold_px': float(threshold_px),
            'center_threshold_cm': (
                float(threshold_px * unit_scale) if calibrated else None
            ),
            'distance_unit': result.distance_unit,
        }

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

        if self.arena_mode:
            headers.extend(ARENA_METRIC_HEADERS)

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

        if self.arena_mode:
            row.extend(self._arena_batch_cells(result))

        return row

    @staticmethod
    def _arena_batch_cells(result: FreestyleAnalysisResult) -> List[any]:
        """Formatted arena columns; N/A when no arena metrics exist."""
        metrics = result._arena_metrics
        if metrics is None:
            return [NOT_AVAILABLE] * len(ARENA_METRIC_HEADERS)

        def number(name, spec):
            value = metrics.get(name)
            return NOT_AVAILABLE if value is None else format(value, spec)

        return [
            number('center_time_s', '.2f'),
            number('periphery_time_s', '.2f'),
            NOT_AVAILABLE if metrics['center_entries'] is None
            else metrics['center_entries'],
            number('latency_to_center_s', '.2f'),
            number('thigmotaxis_time_pct', '.2f'),
            number('thigmotaxis_distance_pct', '.2f'),
            number('mean_wall_distance', '.2f'),
            number('center_speed_mean', '.2f'),
            number('periphery_speed_mean', '.2f'),
        ]
    
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
            FigureCanvasAgg(fig)
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
            FigureCanvasAgg(fig)
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
            FigureCanvasAgg(fig)
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

    def _export_arena_csv(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        """Per-frame arena zone and signed wall distance (positive inside)."""
        try:
            import csv
            unit = result.distance_unit
            with open(
                output_path, 'w', newline='', encoding='utf-8'
            ) as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame', 'timestamp_s', 'arena_zone',
                    f"wall_distance_{unit}", 'distance_unit'
                ])
                for timestamp, label, distance in zip(
                    result.timestamps,
                    result._arena_labels,
                    result._arena_wall_distances,
                ):
                    writer.writerow([
                        int(round(timestamp * result.fps)),
                        f"{timestamp:.3f}",
                        label,
                        '' if distance is None else f"{distance:.2f}",
                        unit,
                    ])
        except Exception as e:
            logger.error(f"Error exporting arena zone CSV: {e}")
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
            if self.arena_mode and result._arena_labels:
                self._export_arena_csv(
                    result, output_dir / f"{base_name}_arena_zones.csv"
                )
    
    def set_immobility_threshold(self, threshold: float) -> None:
        """Set the immobility speed threshold in cm/s (calibrated runs only)."""
        if threshold < 0: raise ValueError("Threshold must be non-negative")
        self.immobility_threshold_cm_per_s = float(threshold)

    def get_algorithm_parameters(self) -> Dict[str, any]:
        parameters = {
            'zones': [list(zone) for zone in self.user_defined_zones],
            'zones_configured': self._zones_configured,
            'immobility_threshold_cm_per_s': self.immobility_threshold_cm_per_s,
            'speed_window_seconds': self.speed_window_seconds,
            'min_immobility_seconds': self.min_immobility_seconds,
            'min_dwell_seconds': self.min_dwell_seconds
        }
        # Written only in arena mode so configurations without an arena keep
        # their exact parameters (and checkpoint settings hash).
        if self.arena_mode:
            parameters.update({
                'arena_mode': True,
                'arena_center_mode': self.arena_center_mode,
                'arena_center_distance_cm': self.arena_center_distance_cm,
                'arena_center_area_fraction': self.arena_center_area_fraction,
            })
        return parameters

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
        # Arena parameters are omitted unless arena mode is on, so a saved
        # configuration without them means "no arena".
        self.configure_arena(
            bool(parameters.get('arena_mode', False)),
            parameters.get('arena_center_mode'),
            parameters.get('arena_center_distance_cm'),
            parameters.get('arena_center_area_fraction'),
        )
