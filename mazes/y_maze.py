"""
Contains the maze-specific logic for the Y-Maze Spontaneous Alternation task.
Thread-safe plotting implementation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type
import logging

# Thread-safe plotting imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
import numpy as np

from mazes.base_maze import BaseAnalysisResult, Maze

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
        canvas = FigureCanvasAgg(fig)
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
        canvas = FigureCanvasAgg(fig)
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

# --- CLASS IMPLEMENTATIONS ---

class _YMazeSequenceTracker:
    """Internal helper to track arm entry sequence for a Y-Maze."""
    def __init__(self):
        self.sequence: List[str] = []
        self.entry_timestamps: List[float] = []
        self.full_sequence: List[str] = []
        self.definitive_zone: str | None = None

    def _is_exclusive_arm(self, zones: List[str]) -> str | None:
        is_arm = any(z.startswith('arm_') for z in zones)
        is_center = 'center' in zones
        if is_arm and not is_center and len(zones) == 1:
            return zones[0]
        return None

    def _is_exclusive_center(self, zones: List[str]) -> bool:
        return zones == ['center']

    def update(self, zones: List[str], timestamp: float):
        exclusive_arm = self._is_exclusive_arm(zones)
        exclusive_center = self._is_exclusive_center(zones)
        if self.definitive_zone is None:
            if exclusive_arm:
                self.definitive_zone = exclusive_arm
                self.full_sequence.append(exclusive_arm)
                self.sequence.append(exclusive_arm.split('_')[-1].upper())
                self.entry_timestamps.append(timestamp)
            elif exclusive_center:
                self.definitive_zone = 'center'
            return

        if self.definitive_zone == 'center' and exclusive_arm:
            self.definitive_zone = exclusive_arm
            self.full_sequence.append(exclusive_arm)
            self.sequence.append(exclusive_arm.split('_')[-1].upper())
            self.entry_timestamps.append(timestamp)
        elif self.definitive_zone.startswith('arm_') and exclusive_center:
            self.definitive_zone = 'center'

    def calculate_alternation(self) -> Tuple[float, int, int]:
        if len(self.sequence) < 3: return 0.0, 0, 0
        correct = sum(1 for i in range(len(self.sequence) - 2) if len(set(self.sequence[i:i + 3])) == 3)
        total_possible = len(self.sequence) - 2
        return (correct / total_possible * 100) if total_possible > 0 else 0.0, correct, total_possible

    def calculate_same_arm_returns(self) -> Tuple[float, int]:
        if len(self.sequence) < 2: return 0.0, 0
        returns = sum(1 for i in range(len(self.sequence) - 1) if self.sequence[i] == self.sequence[i + 1])
        total_transitions = len(self.sequence) - 1
        return (returns / total_transitions * 100) if total_transitions > 0 else 0.0, returns

    def get_sequence_string(self) -> str:
        return "".join(self.sequence)


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


class YMaze(Maze):
    name = "Y-Maze"

    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        return [('arm_a', "Arm A"), ('arm_b', "Arm B"), ('arm_c', "Arm C"), ('center', "Center")]

    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return YMazeAnalysisResult

    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, YMazeAnalysisResult):
            raise TypeError("YMaze logic requires a YMazeAnalysisResult instance.")

        # --- HYSTERESIS LOGIC FOR VISUAL TIMELAPSE AND PLOTTING ---
        definitive_zone = 'outside'
        result.visual_labels = []

        for i, zones in enumerate(result.overlapping_zones):
            # Only change state if cleanly inside a single zone (excluding 'outside')
            if len(zones) == 1 and zones[0] != 'outside':
                definitive_zone = zones[0]
            
            # If in an overlap (len(zones) > 1), definitive_zone retains its previous state automatically!
            result.visual_labels.append(definitive_zone)
        
        # --- EXISTING STRICT SEQUENCE TRACKER LOGIC ---
        tracker = _YMazeSequenceTracker()
        for i, timestamp in enumerate(result.timestamps):
            zones = result.overlapping_zones[i]
            tracker.update(zones, timestamp)

        result.arm_sequence = tracker.get_sequence_string()
        result.arm_entries = list(zip(tracker.full_sequence, tracker.entry_timestamps))
        result.total_arm_entries = len(tracker.sequence)

        alt_pct, alt_correct, alt_total = tracker.calculate_alternation()
        result.alternation_score, result.correct_alternations, result.possible_alternations = alt_pct, alt_correct, alt_total

        sar_pct, sar_count = tracker.calculate_same_arm_returns()
        result.same_arm_returns, result.same_arm_return_count = sar_pct, sar_count

    def get_batch_summary_headers(self) -> List[str]:
        return [
            'arm_sequence', 'alternation_score_pct', 'correct_alternations',
            'possible_alternations', 'total_arm_entries', 'same_arm_return_pct', 'same_arm_return_count'
        ]

    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[any]:
        if not isinstance(result, YMazeAnalysisResult): return ['N/A'] * 7
        return [
            result.arm_sequence,
            f"{result.alternation_score:.2f}",
            result.correct_alternations,
            result.possible_alternations,
            result.total_arm_entries,
            f"{result.same_arm_returns:.2f}",
            result.same_arm_return_count
        ]

    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        if not isinstance(result, YMazeAnalysisResult): return
        _generate_sequence_plot(result, output_dir / f"{base_name}_sequence_plot.png")
        _generate_transition_matrix(result, output_dir / f"{base_name}_transition_matrix.png")