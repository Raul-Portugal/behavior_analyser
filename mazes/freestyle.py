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

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS FOR MOVEMENT ANALYSIS
# ============================================================================

def calculate_instantaneous_speeds(positions: List[Optional[Tuple[int, int]]], 
                                   fps: float) -> List[float]:
    speeds = []
    for i in range(len(positions) - 1):
        if positions[i] is not None and positions[i + 1] is not None:
            dx = positions[i + 1][0] - positions[i][0]
            dy = positions[i + 1][1] - positions[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance * fps  # pixels per second
            speeds.append(speed)
        else:
            speeds.append(0.0)
    
    if speeds:
        speeds.append(speeds[-1])
    else:
        speeds.append(0.0)
    
    return speeds


def calculate_movement_efficiency(positions: List[Optional[Tuple[int, int]]], 
                                  total_distance: float) -> float:
    valid_positions = [p for p in positions if p is not None]
    
    if len(valid_positions) < 2 or total_distance == 0:
        return 0.0
    
    start = valid_positions[0]
    end = valid_positions[-1]
    straight_line = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    efficiency = straight_line / total_distance if total_distance > 0 else 0.0
    return min(efficiency, 1.0)


def calculate_immobility_time(speeds: List[float], fps: float, 
                               threshold: float = 2.0) -> Tuple[float, float]:
    if not speeds:
        return 0.0, 0.0
    
    immobile_frames = sum(1 for s in speeds if s < threshold)
    immobility_time = immobile_frames / fps
    immobility_percentage = (immobile_frames / len(speeds)) * 100
    
    return immobility_time, immobility_percentage


def calculate_zone_transitions(roi_labels: List[str]) -> Tuple[List[Tuple[str, str, int]], int]:
    transitions = []
    total_transitions = 0
    
    if len(roi_labels) < 2:
        return transitions, 0
    
    current_zone = roi_labels[0]
    
    for i in range(1, len(roi_labels)):
        if roi_labels[i] != current_zone:
            transitions.append((current_zone, roi_labels[i], i))
            current_zone = roi_labels[i]
            total_transitions += 1
    
    return transitions, total_transitions


def calculate_zone_first_entries(roi_labels: List[str], 
                                 timestamps: List[float]) -> Dict[str, float]:
    first_entries = {}
    for label, timestamp in zip(roi_labels, timestamps):
        if label != 'outside' and label not in first_entries:
            first_entries[label] = timestamp
    return first_entries


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
    # Movement metrics
    instantaneous_speeds: List[float] = field(default_factory=list)
    average_speed: float = 0.0
    peak_speed: float = 0.0
    immobility_time: float = 0.0
    immobility_percentage: float = 0.0
    movement_efficiency: float = 0.0
    
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
                'movement_efficiency': self.movement_efficiency
            },
            'exploration': {
                'total_transitions': self.total_transitions,
                'zone_entry_counts': self.zone_entry_counts,
                'zone_first_entries': self.zone_first_entries
            },
            'custom': self.custom_metrics
        }
        
        if len(self.instantaneous_speeds) < 10000:
            base_dict['freestyle_metrics']['instantaneous_speeds'] = self.instantaneous_speeds
        
        return base_dict


# ============================================================================
# FREESTYLE MAZE CLASS
# ============================================================================

class Freestyle(Maze):
    """Freestyle/Open Field maze type with flexible zone configuration."""
    
    def __init__(self):
        self.user_defined_zones: List[Tuple[str, str]] = []
        self._zones_configured = False
        self._immobility_threshold = 2.0  # pixels/second
    
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
    
    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, FreestyleAnalysisResult):
            raise TypeError("Freestyle logic requires a FreestyleAnalysisResult instance.")
        
        logger.info("Calculating Freestyle metrics...")
        
        # Movement metrics
        result.instantaneous_speeds = calculate_instantaneous_speeds(
            result.positions, result.fps
        )
        
        valid_speeds = [s for s in result.instantaneous_speeds if s > 0]
        result.average_speed = np.mean(valid_speeds) if valid_speeds else 0.0
        result.peak_speed = np.max(valid_speeds) if valid_speeds else 0.0
        
        result.immobility_time, result.immobility_percentage = calculate_immobility_time(
            result.instantaneous_speeds, result.fps, self._immobility_threshold
        )
        
        result.movement_efficiency = calculate_movement_efficiency(
            result.positions, result.total_distance
        )
        
        # Exploration metrics
        if self.user_defined_zones:
            result.zone_transitions, result.total_transitions = calculate_zone_transitions(
                result.roi_labels
            )
            for from_zone, to_zone, _ in result.zone_transitions:
                result.zone_entry_counts[to_zone] = result.zone_entry_counts.get(to_zone, 0) + 1
            result.zone_first_entries = calculate_zone_first_entries(
                result.roi_labels, result.timestamps
            )
        
        logger.info("Freestyle metrics calculation complete")
    
    def get_batch_summary_headers(self) -> List[str]:
        headers = [
            'average_speed', 'peak_speed', 
            'immobility_time_s', 'immobility_pct',
            'movement_efficiency'
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
        
        row = [
            f"{result.average_speed:.2f}",
            f"{result.peak_speed:.2f}",
            f"{result.immobility_time:.2f}",
            f"{result.immobility_percentage:.2f}",
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
        self._export_speed_csv(result, output_dir / f"{base_name}_speed_data.csv")
        
        if self.user_defined_zones and result.total_transitions > 0:
            self._generate_transition_matrix_plot(result, output_dir / f"{base_name}_transition_matrix.png")
            self._generate_zone_entries_plot(result, output_dir / f"{base_name}_zone_entries.png")
    
    def _generate_speed_plot(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        try:
            if not result.instantaneous_speeds or not result.timestamps: return
            
            # Thread-safe plotting
            fig = Figure(figsize=(14, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            ax.plot(result.timestamps, result.instantaneous_speeds, 
                   'b-', alpha=0.6, linewidth=0.5, label='Instantaneous Speed')
            
            ax.axhline(y=result.average_speed, color='g', linestyle='--', 
                      linewidth=2, label=f'Average: {result.average_speed:.1f} px/s')
            
            ax.axhline(y=self._immobility_threshold, color='r', linestyle='--', 
                      linewidth=1, alpha=0.5, label=f'Immobility: {self._immobility_threshold} px/s')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Speed (pixels/second)')
            ax.set_title('Movement Speed Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            
        except Exception as e:
            logger.error(f"Error generating speed plot: {e}")
    
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
    
    def _export_speed_csv(self, result: FreestyleAnalysisResult, output_path: Path) -> None:
        try:
            import csv
            if not result.instantaneous_speeds or not result.timestamps: return
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                speed_unit = f"speed_{result.distance_unit}_per_s"
                writer.writerow(['frame', 'timestamp_s', speed_unit, 'is_immobile', 'distance_unit'])
                
                for i, (timestamp, speed) in enumerate(zip(result.timestamps, result.instantaneous_speeds)):
                    is_immobile = 1 if speed < self._immobility_threshold else 0
                    writer.writerow([
                        result.start_frame + i,
                        f"{timestamp:.3f}",
                        f"{speed:.2f}",
                        is_immobile,
                        result.distance_unit
                    ])
        except Exception as e:
            logger.error(f"Error exporting speed CSV: {e}")
    
    def set_immobility_threshold(self, threshold: float) -> None:
        if threshold < 0: raise ValueError("Threshold must be non-negative")
        self._immobility_threshold = threshold