"""
Logic for Tail Suspension Test (TST).
Uses rigid/articulated motion decomposition with per-mouse timing support.
"""
from dataclasses import dataclass, field
from bisect import bisect_left
from pathlib import Path
from typing import List, Tuple, Type, Dict, Optional
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from mazes.base_maze import BaseAnalysisResult, Maze
from core.tst_events import smooth_tst_immobility_states
from core.tst_motion_engine import TSTMotionConfig, TSTMotionEngine

@dataclass
class TSTResult(BaseAnalysisResult):
    activity_scores: Dict[str, List[float]] = field(default_factory=dict)
    motion_evidence: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    behavior_states: Dict[str, List[str]] = field(default_factory=dict)
    state_confidences: Dict[str, List[float]] = field(default_factory=dict)
    immobile_states: Dict[str, List[Optional[bool]]] = field(default_factory=dict)
    per_roi_times: Optional[Dict[str, Tuple[float, float]]] = None
    
    total_immobility_time: Dict[str, float] = field(default_factory=dict)
    immobility_bouts: Dict[str, int] = field(default_factory=dict)
    latency_to_first_immobility: Dict[str, Optional[float]] = field(default_factory=dict)
    static_immobility_time: Dict[str, float] = field(default_factory=dict)
    passive_swing_time: Dict[str, float] = field(default_factory=dict)
    bridged_gap_time: Dict[str, float] = field(default_factory=dict)
    mobile_time: Dict[str, float] = field(default_factory=dict)
    uncertain_time: Dict[str, float] = field(default_factory=dict)
    valid_time: Dict[str, float] = field(default_factory=dict)
    immobility_percentage: Dict[str, float] = field(default_factory=dict)
    mean_state_confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "metadata": {
                "analysis_type": TSTMotionEngine.ENGINE_ID,
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "total_frames": len(self.timestamps),
                "fps": self.fps,
                "per_roi_times": self.per_roi_times or {},
            },
            "tst_metrics": {
                "total_immobility_time_s": self.total_immobility_time,
                "immobility_percentage": self.immobility_percentage,
                "static_immobility_time_s": self.static_immobility_time,
                "passive_swing_time_s": self.passive_swing_time,
                "bridged_gap_time_s": self.bridged_gap_time,
                "mobile_time_s": self.mobile_time,
                "uncertain_time_s": self.uncertain_time,
                "valid_time_s": self.valid_time,
                "mean_state_confidence": self.mean_state_confidence,
                "immobility_bouts": self.immobility_bouts,
                "latency_to_first_immobility_s": (
                    self.latency_to_first_immobility
                ),
            },
        }

class TST(Maze):
    name = "Tail Suspension Test"
    
    def __init__(self):
        self.num_mice = 1
        self._custom_rois = []
        self.min_time_seconds = 1.0
        self.max_motion_gap_seconds = 0.20
        self.motion_config = TSTMotionConfig()

    def configure_mice(self, count: int):
        self.num_mice = count
        self._custom_rois = [(f"mouse_{i+1}", f"Mouse {i+1}") for i in range(count)]

    def get_roi_definitions(self) -> List[Tuple[str, str]]:
        if not self._custom_rois: return [("mouse_1", "Mouse 1")]
        return self._custom_rois

    def get_result_class(self) -> Type[BaseAnalysisResult]:
        return TSTResult

    def set_parameters(
        self,
        min_time_seconds: float,
        max_motion_gap_seconds: Optional[float] = None
    ):
        self.min_time_seconds = max(0.0, min_time_seconds)
        if max_motion_gap_seconds is not None:
            self.max_motion_gap_seconds = max(0.0, max_motion_gap_seconds)

    def get_algorithm_parameters(self) -> Dict[str, object]:
        return {
            'engine_id': TSTMotionEngine.ENGINE_ID,
            'num_mice': self.num_mice,
            'min_time_seconds': self.min_time_seconds,
            'max_motion_gap_seconds': self.max_motion_gap_seconds,
            'motion_config': dict(self.motion_config.__dict__),
        }

    def configure_algorithm_parameters(self, parameters: Dict[str, object]) -> None:
        if not parameters:
            return
        engine_id = parameters.get("engine_id")
        if engine_id != TSTMotionEngine.ENGINE_ID:
            raise ValueError(
                "TST parameters are not compatible with "
                f"{TSTMotionEngine.ENGINE_ID}; re-tune and save a new "
                "configuration"
            )
        self.configure_mice(int(parameters.get('num_mice', self.num_mice)))
        motion_values = parameters.get('motion_config', {})
        if isinstance(motion_values, dict):
            valid_keys = TSTMotionConfig.__annotations__.keys()
            self.motion_config = TSTMotionConfig(**{
                key: value
                for key, value in motion_values.items()
                if key in valid_keys
            })
            self.motion_config.validate()
        self.set_parameters(
            float(parameters.get('min_time_seconds', self.min_time_seconds)),
            float(parameters.get(
                'max_motion_gap_seconds', self.max_motion_gap_seconds
            ))
        )

    def calculate_metrics(self, result: BaseAnalysisResult) -> None:
        if not isinstance(result, TSTResult): return
        
        fps = result.fps
        min_frames = max(1, int(round(self.min_time_seconds * fps)))
        max_gap_frames = max(0, int(round(self.max_motion_gap_seconds * fps)))
        mouse_names = sorted(
            set(result.motion_evidence)
            | set(result.activity_scores)
            | set(result.per_roi_times or {})
        )
        for mouse_name in mouse_names:
            evidence_trace = result.motion_evidence.get(mouse_name, [])
            if not evidence_trace:
                result.total_immobility_time[mouse_name] = 0.0
                result.immobility_percentage[mouse_name] = 0.0
                result.static_immobility_time[mouse_name] = 0.0
                result.passive_swing_time[mouse_name] = 0.0
                result.bridged_gap_time[mouse_name] = 0.0
                result.mobile_time[mouse_name] = 0.0
                result.uncertain_time[mouse_name] = 0.0
                result.valid_time[mouse_name] = 0.0
                result.mean_state_confidence[mouse_name] = 0.0
                result.immobility_bouts[mouse_name] = 0
                result.latency_to_first_immobility[mouse_name] = None
                result.immobile_states[mouse_name] = []
                result.behavior_states[mouse_name] = []
                result.state_confidences[mouse_name] = []
                continue
            
            # --- TIMING LOGIC ---
            start_idx = 0
            end_idx = len(evidence_trace)
            
            if result.per_roi_times and mouse_name in result.per_roi_times:
                target_start, target_end = result.per_roi_times[mouse_name]
                trace_times = result.timestamps[:len(evidence_trace)]
                start_idx = bisect_left(trace_times, target_start)
                end_idx = bisect_left(trace_times, target_end)
            
            valid_evidence = evidence_trace[start_idx:end_idx]
            if not valid_evidence:
                result.total_immobility_time[mouse_name] = 0.0
                result.immobility_percentage[mouse_name] = 0.0
                result.static_immobility_time[mouse_name] = 0.0
                result.passive_swing_time[mouse_name] = 0.0
                result.bridged_gap_time[mouse_name] = 0.0
                result.mobile_time[mouse_name] = 0.0
                result.uncertain_time[mouse_name] = 0.0
                result.valid_time[mouse_name] = 0.0
                result.mean_state_confidence[mouse_name] = 0.0
                result.immobility_bouts[mouse_name] = 0
                result.latency_to_first_immobility[mouse_name] = None
                result.immobile_states[mouse_name] = [None] * len(
                    evidence_trace
                )
                result.behavior_states[mouse_name] = [
                    str(item.get("state", "uncertain"))
                    for item in evidence_trace
                ]
                result.state_confidences[mouse_name] = [
                    float(item.get("confidence", 0.0))
                    for item in evidence_trace
                ]
                continue

            states = [
                str(item.get("state", "uncertain"))
                for item in valid_evidence
            ]
            confidences = [
                float(item.get("confidence", 0.0))
                for item in valid_evidence
            ]
            filtered_immobility = smooth_tst_immobility_states(
                states, min_frames, max_gap_frames
            )
            
            # Pad state for alignment
            full_state: List[Optional[bool]] = [None] * len(evidence_trace)
            if len(filtered_immobility) > 0:
                full_state[
                    start_idx:start_idx + len(filtered_immobility)
                ] = filtered_immobility
                
            result.immobile_states[mouse_name] = full_state
            result.behavior_states[mouse_name] = [
                str(item.get("state", "uncertain"))
                for item in evidence_trace
            ]
            result.state_confidences[mouse_name] = [
                float(item.get("confidence", 0.0))
                for item in evidence_trace
            ]
            result.mean_state_confidence[mouse_name] = (
                float(np.mean(confidences)) if confidences else 0.0
            )
            
            immobile_count = sum(value is True for value in filtered_immobility)
            result.total_immobility_time[mouse_name] = immobile_count / fps
            valid_count = sum(value is not None for value in filtered_immobility)
            result.valid_time[mouse_name] = valid_count / fps
            result.immobility_percentage[mouse_name] = (
                immobile_count / valid_count * 100.0 if valid_count else 0.0
            )
            result.static_immobility_time[mouse_name] = sum(
                value is True and state == "immobile_static"
                for value, state in zip(filtered_immobility, states)
            ) / fps
            result.passive_swing_time[mouse_name] = sum(
                value is True and state == "passive_swing"
                for value, state in zip(filtered_immobility, states)
            ) / fps
            result.bridged_gap_time[mouse_name] = sum(
                value is True and state == "active_escape"
                for value, state in zip(filtered_immobility, states)
            ) / fps
            result.mobile_time[mouse_name] = sum(
                value is False for value in filtered_immobility
            ) / fps
            result.uncertain_time[mouse_name] = (
                sum(
                    state not in {
                        "immobile_static", "passive_swing", "active_escape"
                    }
                    for state in states
                ) / fps
            )
            
            bouts = 0
            previous = False
            for value in filtered_immobility:
                current = value is True
                if current and not previous:
                    bouts += 1
                previous = current
            result.immobility_bouts[mouse_name] = bouts
            
            first_idx = next(
                (
                    index for index, value in enumerate(filtered_immobility)
                    if value is True
                ),
                None,
            )
            if first_idx is not None:
                # Latency relative to Analysis Window Start (not video start)
                result.latency_to_first_immobility[mouse_name] = first_idx / fps
            else:
                result.latency_to_first_immobility[mouse_name] = None

    def get_batch_summary_headers(self) -> List[str]:
        return [
            'mouse_name', 'total_immobility_s', 'immobility_pct',
            'static_immobility_s', 'passive_swing_s',
            'bridged_gap_immobility_s', 'mobile_s', 'uncertain_s',
            'valid_time_s', 'bouts', 'latency_s',
            'mean_state_confidence', 'window_start', 'window_end'
        ]

    def get_batch_summary_row(self, result: BaseAnalysisResult) -> List[List[any]]:
        """Returns LIST OF ROWS (one per mouse)."""
        if not isinstance(result, TSTResult): return []
        
        rows = []
        mice = sorted(result.total_immobility_time.keys())
        for mouse in mice:
            start_t = (
                result.timestamps[0]
                if result.timestamps else result.start_frame / result.fps
            )
            end_t = (
                result.timestamps[-1] + (1.0 / result.fps)
                if result.timestamps and result.fps > 0 else
                result.end_frame / result.fps
            )
            if result.per_roi_times and mouse in result.per_roi_times:
                start_t, end_t = result.per_roi_times[mouse]
            
            total = result.total_immobility_time.get(mouse, 0)
            bouts = result.immobility_bouts.get(mouse, 0)
            lat = result.latency_to_first_immobility.get(mouse)
            
            row = [
                mouse,
                f"{total:.2f}",
                f"{result.immobility_percentage.get(mouse, 0.0):.2f}",
                f"{result.static_immobility_time.get(mouse, 0.0):.2f}",
                f"{result.passive_swing_time.get(mouse, 0.0):.2f}",
                f"{result.bridged_gap_time.get(mouse, 0.0):.2f}",
                f"{result.mobile_time.get(mouse, 0.0):.2f}",
                f"{result.uncertain_time.get(mouse, 0.0):.2f}",
                f"{result.valid_time.get(mouse, 0.0):.2f}",
                bouts,
                "" if lat is None else f"{lat:.2f}",
                f"{result.mean_state_confidence.get(mouse, 0.0):.4f}",
                f"{start_t:.1f}",
                f"{end_t:.1f}"
            ]
            rows.append(row)
        return rows

    def generate_specific_plots(self, result: BaseAnalysisResult, output_dir: Path, base_name: str) -> None:
        if not isinstance(result, TSTResult) or not result.activity_scores: return
        
        num_mice = len(result.activity_scores)
        if num_mice == 0: return

        fig = Figure(figsize=(12, 4 * num_mice))
        canvas = FigureCanvasAgg(fig)
        axs = fig.subplots(num_mice, 1, squeeze=False)
        
        timestamps = result.timestamps
        
        for i, (name, activity) in enumerate(result.activity_scores.items()):
            ax = axs[i, 0]
            plot_len = min(len(timestamps), len(activity))
            t_plot = timestamps[:plot_len]
            e_plot = activity[:plot_len]
            
            if plot_len == 0: continue

            if result.per_roi_times and name in result.per_roi_times:
                s, e = result.per_roi_times[name]
                ax.axvspan(s, e, color='green', alpha=0.05)
                # Gray out excluded
                if s > t_plot[0]: ax.axvspan(t_plot[0], s, color='gray', alpha=0.2)
                if e < t_plot[-1]: ax.axvspan(e, t_plot[-1], color='gray', alpha=0.2)

            ax.plot(t_plot, e_plot, 'k-', alpha=0.5, linewidth=0.5)
            
            state_values = result.immobile_states.get(
                name, [None] * plot_len
            )[:plot_len]
            immobile = [value is True for value in state_values]
            uncertain = [value is None for value in state_values]
            max_val = max(e_plot) if e_plot else 1.0
            ax.fill_between(t_plot, 0, max_val * 1.1, 
                           where=immobile, color='red', alpha=0.3, label='Immobile')
            ax.fill_between(
                t_plot, 0, max_val * 1.1,
                where=uncertain, color='orange', alpha=0.2,
                label='Uncertain'
            )
            
            ax.axhline(y=50.0, color='b', linestyle='--',
                       label='Active-evidence reference')
            ax.set_title(f"{name.replace('_', ' ').title()}")
            ax.set_ylabel("Active-motion evidence (0-100)")
            ax.legend(loc="upper right")
            
        fig.tight_layout()
        fig.savefig(output_dir / f"{base_name}_tst_ethogram.png", dpi=150)
