"""
Contains the maze-specific logic for the Y-Maze Spontaneous Alternation task.
Thread-safe plotting implementation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type
import logging
import csv

# Thread-safe plotting imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
import numpy as np

from mazes.base_maze import BaseAnalysisResult, Maze
from core.zone_events import (
    debounce_zone_labels,
    dwell_frames,
    analyze_center_mediated_entries,
    overlapping_zones_to_candidates,
)

logger = logging.getLogger(__name__)

# --- HELPER PLOTTING FUNCTIONS ---

def _generate_sequence_plot(result: 'YMazeAnalysisResult', output_path: Path):
    """Generates a plot of the arm entry sequence, specific to Y-Maze."""
    logger.info("Generating Y-Maze arm entry sequence plot...")
    sequence = result.arm_sequence
    if not sequence: return
    
    try:
        arm_map = {'A': 0, 'B': 1, 'C': 2}
        x = list(range(len(sequence)))
        y = [arm_map.get(arm, -1) for arm in sequence]
        
        colors = ['blue'] * len(sequence)
        for i in range(len(sequence)):
            if i > 1 and len(set(sequence[i - 2:i + 1])) == 3: colors[i] = 'green'
            elif i > 0 and sequence[i] == sequence[i - 1]: colors[i] = 'red'
        
        # Use Figure object
        fig = Figure(figsize=(14, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(x, y, 'k-', alpha=0.3, linewidth=1, zorder=2)
        ax.scatter(x, y, c=colors, s=100, alpha=0.7, zorder=3, edgecolors='k')
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Arm A', 'Arm B', 'Arm C'])
        ax.set_xlabel('Entry Number')
        ax.set_title(f'Arm Entry Sequence: {sequence}')
        
        legend_elements = [
            Patch(facecolor='green', label='Correct Alternation'),
            Patch(facecolor='red', label='Same Arm Return'),
            Patch(facecolor='blue', label='Other Transition')
        ]
        ax.legend(handles=legend_elements)
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    except Exception as e:
        logger.error(f"Failed to generate sequence plot: {e}")
        raise

def _generate_transition_matrix(result: 'YMazeAnalysisResult', output_path: Path):
    """Generates a heatmap of the arm-to-arm transition counts."""
    logger.info("Generating arm transition matrix...")
    if not hasattr(result, 'arm_entries') or len(result.arm_entries) < 2: return
    
    try:
        sequence = result.arm_entries
        arm_names = sorted(list(set(arm for arm, ts in sequence)))
        if not arm_names: return
        
        matrix = np.zeros((len(arm_names), len(arm_names)), dtype=int)
        arm_to_idx = {name: i for i, name in enumerate(arm_names)}
        
        for i in range(len(sequence) - 1):
            from_arm, _ = sequence[i]
            to_arm, _ = sequence[i + 1]
            from_idx = arm_to_idx.get(from_arm)
            to_idx = arm_to_idx.get(to_arm)
            if from_idx is not None and to_idx is not None: 
                matrix[from_idx, to_idx] += 1
        
        fig = Figure(figsize=(8, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        # Manual heatmap using imshow (thread-safe)
        im = ax.imshow(matrix, cmap='Blues')
        
        # Add counts
        for i in range(len(arm_names)):
            for j in range(len(arm_names)):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

        # Labels
        labels = [name.replace('_', ' ').title() for name in arm_names]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        fig.colorbar(im, ax=ax, label='Transition Count')
        ax.set_xlabel('To Arm')
        ax.set_ylabel('From Arm')
        ax.set_title('Arm Transition Matrix')
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    except Exception as e:
        logger.error(f"Failed to generate transition matrix: {e}")
        raise

# --- CLASS IMPLEMENTATIONS ---

@dataclass
class YMazeAnalysisResult(BaseAnalysisResult):
    """Y-Maze specific result fields."""
    arm_sequence: str = ""
    arm_entries: List[Tuple[str, float]] = field(default_factory=list)
    alternation_score: float = 0.0
    correct_alternations: int = 0
    possible_alternations: int = 0
    same_arm_returns: float = 0.0
    same_arm_return_count: int = 0
    total_arm_entries: int = 0
    # Arm entries a tracking gap may have hidden (not included in the counts).
    entries_possibly_missed: int = 0


class YMaze(Maze):
    name = "Y-Maze"

    def __init__(self):
        self.min_dwell_seconds = 0.10
        self.count_initial_arm = True

    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        return [('arm_a', "Arm A"), ('arm_b', "Arm B"), ('arm_c', "Arm C"), ('center', "Center")]

    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return YMazeAnalysisResult

    def resolve_tracking_labels(self, result: BaseAnalysisResult) -> None:
        candidates = overlapping_zones_to_candidates(result.overlapping_zones)
        resolved = debounce_zone_labels(
            candidates,
            dwell_frames(self.min_dwell_seconds, result.fps)
        )
        result.roi_labels = resolved
        result.visual_labels = list(resolved)

    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, YMazeAnalysisResult):
            raise TypeError("YMaze logic requires a YMazeAnalysisResult instance.")

        arm_labels = {'arm_a', 'arm_b', 'arm_c'}
        counted = analyze_center_mediated_entries(
            result.roi_labels,
            result.timestamps,
            arm_labels=arm_labels,
            count_initial_arm=self.count_initial_arm
        )
        result.arm_entries = counted.entries
        result.entries_possibly_missed = counted.possibly_missed
        sequence = [
            arm.split('_')[-1].upper()
            for arm, _ in result.arm_entries
        ]
        result.arm_sequence = "".join(sequence)
        result.total_arm_entries = len(sequence)

        result.possible_alternations = max(0, len(sequence) - 2)
        result.correct_alternations = sum(
            len(set(sequence[index:index + 3])) == 3
            for index in range(result.possible_alternations)
        )
        result.alternation_score = (
            result.correct_alternations / result.possible_alternations * 100
            if result.possible_alternations else 0.0
        )

        total_transitions = max(0, len(sequence) - 1)
        result.same_arm_return_count = sum(
            sequence[index] == sequence[index + 1]
            for index in range(total_transitions)
        )
        result.same_arm_returns = (
            result.same_arm_return_count / total_transitions * 100
            if total_transitions else 0.0
        )

    def get_algorithm_parameters(self) -> dict:
        return {
            'min_dwell_seconds': self.min_dwell_seconds,
            'count_initial_arm': self.count_initial_arm
        }

    def configure_algorithm_parameters(self, parameters: dict) -> None:
        if not parameters:
            return
        self.min_dwell_seconds = float(
            parameters.get('min_dwell_seconds', self.min_dwell_seconds)
        )
        self.count_initial_arm = bool(
            parameters.get('count_initial_arm', self.count_initial_arm)
        )

    def get_batch_summary_headers(self) -> List[str]:
        return [
            'arm_sequence', 'alternation_score_pct', 'correct_alternations',
            'possible_alternations', 'total_arm_entries', 'same_arm_return_pct', 'same_arm_return_count',
            'entries_possibly_missed'
        ]

    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[any]:
        if not isinstance(result, YMazeAnalysisResult): return ['N/A'] * 8
        return [
            result.arm_sequence,
            f"{result.alternation_score:.2f}",
            result.correct_alternations,
            result.possible_alternations,
            result.total_arm_entries,
            f"{result.same_arm_returns:.2f}",
            result.same_arm_return_count,
            result.entries_possibly_missed
        ]

    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        if not isinstance(result, YMazeAnalysisResult): return
        _generate_sequence_plot(result, output_dir / f"{base_name}_sequence_plot.png")
        _generate_transition_matrix(result, output_dir / f"{base_name}_transition_matrix.png")

    def export_specific_data(
        self,
        result: BaseAnalysisResult,
        output_dir: Path,
        base_name: str
    ) -> None:
        if not isinstance(result, YMazeAnalysisResult):
            return
        output_path = output_dir / f"{base_name}_arm_entries.csv"
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['entry_number', 'arm', 'timestamp_s'])
            for index, (arm, timestamp) in enumerate(result.arm_entries, 1):
                writer.writerow([index, arm, f"{timestamp:.3f}"])
