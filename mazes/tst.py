"""
Logic for Tail Suspension Test (TST).
Focuses on Motion Energy quantification with Per-Mouse Timing support.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type, Dict, Optional
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from mazes.base_maze import BaseAnalysisResult, Maze

@dataclass
class TSTResult(BaseAnalysisResult):
    motion_energy: Dict[str, List[float]] = field(default_factory=dict)
    immobile_states: Dict[str, List[bool]] = field(default_factory=dict)
    per_roi_times: Optional[Dict[str, Tuple[float, float]]] = None
    
    total_immobility_time: Dict[str, float] = field(default_factory=dict)
    immobility_bouts: Dict[str, int] = field(default_factory=dict)
    latency_to_first_immobility: Dict[str, float] = field(default_factory=dict)

class TST(Maze):
    name = "Tail Suspension Test"
    
    def __init__(self):
        self.num_mice = 1
        self._custom_rois = []
        self.energy_threshold = 10.0
        self.min_time_seconds = 1.0

    def configure_mice(self, count: int):
        self.num_mice = count
        self._custom_rois = [(f"mouse_{i+1}", f"Mouse {i+1}") for i in range(count)]

    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        if not self._custom_rois: return [("mouse_1", "Mouse 1")]
        return self._custom_rois

    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return TSTResult

    def set_parameters(self, energy_threshold: float, min_time_seconds: float):
        self.energy_threshold = energy_threshold
        self.min_time_seconds = min_time_seconds

    def _apply_temporal_filter(self, binary_array: np.ndarray, min_frames: int) -> np.ndarray:
        n = len(binary_array)
        filtered = np.zeros(n, dtype=bool)
        current_bout_start = -1
        
        for i in range(n):
            if binary_array[i]:
                if current_bout_start == -1: current_bout_start = i
            else:
                if current_bout_start != -1:
                    if (i - current_bout_start) >= min_frames:
                        filtered[current_bout_start:i] = True
                    current_bout_start = -1
        
        if current_bout_start != -1 and (n - current_bout_start) >= min_frames:
            filtered[current_bout_start:] = True
                
        return filtered

    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, TSTResult): return
        
        fps = result.fps
        min_frames = int(self.min_time_seconds * fps)
        video_start_time = result.start_frame / fps
        
        for mouse_name, energy_trace in result.motion_energy.items():
            if not energy_trace: continue
            
            # --- TIMING LOGIC ---
            start_idx = 0
            end_idx = len(energy_trace)
            
            if result.per_roi_times and mouse_name in result.per_roi_times:
                target_start, target_end = result.per_roi_times[mouse_name]
                
                rel_start = max(0.0, target_start - video_start_time)
                rel_end = max(0.0, target_end - video_start_time)
                
                start_idx = int(rel_start * fps)
                end_idx = int(rel_end * fps)
                
                start_idx = max(0, min(start_idx, len(energy_trace)))
                end_idx = max(start_idx, min(end_idx, len(energy_trace)))
            
            valid_energy = np.array(energy_trace[start_idx:end_idx])
            
            if len(valid_energy) == 0:
                result.total_immobility_time[mouse_name] = 0.0
                continue

            raw_immobility = valid_energy < self.energy_threshold
            filtered_immobility = self._apply_temporal_filter(raw_immobility, min_frames)
            
            # Pad state for alignment
            full_state = [False] * len(energy_trace)
            if len(filtered_immobility) > 0:
                full_state[start_idx : start_idx+len(filtered_immobility)] = filtered_immobility.tolist()
                
            result.immobile_states[mouse_name] = full_state
            
            immobile_count = np.sum(filtered_immobility)
            result.total_immobility_time[mouse_name] = immobile_count / fps
            
            changes = np.diff(filtered_immobility.astype(int))
            bouts = np.sum(changes == 1)
            if len(filtered_immobility) > 0 and filtered_immobility[0]: bouts += 1
            result.immobility_bouts[mouse_name] = int(bouts)
            
            if np.any(filtered_immobility):
                first_idx = np.argmax(filtered_immobility)
                # Latency relative to Analysis Window Start (not video start)
                result.latency_to_first_immobility[mouse_name] = first_idx / fps
            else:
                result.latency_to_first_immobility[mouse_name] = 0.0

    def get_batch_summary_headers(self) -> List[str]:
        return ['mouse_name', 'total_immobility_s', 'bouts', 'latency_s', 'window_start', 'window_end']

    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[List[any]]:
        """Returns LIST OF ROWS (one per mouse)."""
        if not isinstance(result, TSTResult): return []
        
        rows = []
        mice = sorted(result.total_immobility_time.keys())
        for mouse in mice:
            start_t, end_t = 0, 0
            if result.per_roi_times and mouse in result.per_roi_times:
                start_t, end_t = result.per_roi_times[mouse]
            
            total = result.total_immobility_time.get(mouse, 0)
            bouts = result.immobility_bouts.get(mouse, 0)
            lat = result.latency_to_first_immobility.get(mouse, 0)
            
            row = [
                mouse,
                f"{total:.2f}",
                bouts,
                f"{lat:.2f}",
                f"{start_t:.1f}",
                f"{end_t:.1f}"
            ]
            rows.append(row)
        return rows

    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        if not isinstance(result, TSTResult) or not result.motion_energy: return
        
        num_mice = len(result.motion_energy)
        if num_mice == 0: return

        fig = Figure(figsize=(12, 4 * num_mice))
        canvas = FigureCanvasAgg(fig)
        axs = fig.subplots(num_mice, 1, squeeze=False)
        
        timestamps = result.timestamps
        
        for i, (name, energy) in enumerate(result.motion_energy.items()):
            ax = axs[i, 0]
            plot_len = min(len(timestamps), len(energy))
            t_plot = timestamps[:plot_len]
            e_plot = energy[:plot_len]
            
            if plot_len == 0: continue

            if result.per_roi_times and name in result.per_roi_times:
                s, e = result.per_roi_times[name]
                ax.axvspan(s, e, color='green', alpha=0.05)
                # Gray out excluded
                if s > t_plot[0]: ax.axvspan(t_plot[0], s, color='gray', alpha=0.2)
                if e < t_plot[-1]: ax.axvspan(e, t_plot[-1], color='gray', alpha=0.2)

            ax.plot(t_plot, e_plot, 'k-', alpha=0.5, linewidth=0.5)
            
            immobile = result.immobile_states.get(name, [False]*plot_len)[:plot_len]
            max_val = max(e_plot) if e_plot else 1.0
            ax.fill_between(t_plot, 0, max_val * 1.1, 
                           where=immobile, color='red', alpha=0.3, label='Immobile')
            
            ax.axhline(y=self.energy_threshold, color='b', linestyle='--')
            ax.set_title(f"{name.replace('_', ' ').title()}")
            
        fig.tight_layout()
        fig.savefig(output_dir / f"{base_name}_tst_ethogram.png", dpi=150)