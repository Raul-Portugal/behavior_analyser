"""Transparent evaluation helpers for independently annotated TST frames.

This module deliberately does not create annotation files during normal runs.
It is for validating a fixed engine/profile against a separately prepared
ground-truth CSV when such data becomes available.
"""
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VALID_STATES = {"immobile_static", "passive_swing", "active_escape", "uncertain"}
IMMOBILE_STATES = frozenset({"immobile_static", "passive_swing"})


def load_annotations(path: Path) -> Dict[Tuple[str, int], str]:
    """Read ``mouse_name,frame,state`` annotations and validate their labels."""
    annotations = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            mouse = str(row["mouse_name"]).strip()
            state = str(row["state"]).strip()
            if state not in VALID_STATES:
                raise ValueError(f"Unsupported TST annotation state: {state}")
            annotations[(mouse, int(row["frame"]))] = state
    return annotations


def evaluate_annotations(
    motion_evidence: Mapping[str, Iterable[Mapping[str, object]]],
    annotations: Mapping[Tuple[str, int], str],
    start_frame: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Return frame-wise agreement and immobility precision/recall/F1 by mouse.

    Annotations are keyed by absolute video frame. Evidence rows carry their
    frame as ``source_frame``; for rows that do not, the frame is
    ``start_frame`` plus the row's position (pass the absolute frame of the
    first evidence row).
    """
    results = {}
    for mouse, evidence in motion_evidence.items():
        pairs = []
        for index, item in enumerate(evidence):
            frame = int(item.get("source_frame", start_frame + index))
            truth = annotations.get((mouse, frame))
            if truth is not None:
                pairs.append((str(item.get("state", "uncertain")), truth))
        if not pairs:
            continue
        confusion = Counter(pairs)
        predicted_immobile = {"immobile_static", "passive_swing"}
        true_positive = sum(p in predicted_immobile and t in predicted_immobile for p, t in pairs)
        false_positive = sum(p in predicted_immobile and t not in predicted_immobile for p, t in pairs)
        false_negative = sum(p not in predicted_immobile and t in predicted_immobile for p, t in pairs)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        results[mouse] = {
            "annotated_frames": float(len(pairs)),
            "exact_state_agreement_pct": sum(p == t for p, t in pairs) / len(pairs) * 100.0,
            "immobility_precision": precision,
            "immobility_recall": recall,
            "immobility_f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "confusion_entries": float(len(confusion)),
        }
    return results


# ---------------------------------------------------------------------------
# Reading the exported motion evidence
# ---------------------------------------------------------------------------

_EVIDENCE_FLOAT_COLUMNS = (
    "confidence", "rigid_motion", "rotation_motion", "rotation_radians",
    "rigid_dx", "rigid_dy", "residual_motion", "deformation", "coherence",
    "motion_injection", "angular_injection", "pendulum_score",
    "mask_area_ratio", "affine_inlier_ratio",
)
_EVIDENCE_FLAG_COLUMNS = ("temporal_ready", "camera_stabilized")
_STATE_SUFFIX = "behavior_state"


def _optional_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    return float(text) if text else None


def load_tst_evidence_csv(path: Path) -> Dict[str, List[Dict[str, object]]]:
    """Read ``<base>_tst_motion_evidence.csv`` back into per-mouse evidence.

    The result has the shape ``evaluate_annotations`` expects: one list of
    row dictionaries per mouse, in frame order. The CSV ``frame`` column (the
    absolute video frame the exporter writes as ``round(timestamp * fps)``)
    becomes ``source_frame``, which the application keeps in memory only.
    Each row also carries ``timestamp_s``, ``state`` (from
    ``<mouse>_behavior_state``), ``is_immobile`` (the smoothed immobility flag;
    ``None`` outside the scored window), ``active_evidence`` (``None`` when
    blank), the numeric motion features, ``temporal_ready``,
    ``camera_stabilized`` and ``qc_flag``.

    Mouse names are taken from the ``<mouse>_behavior_state`` headers and
    may themselves contain underscores.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        header = list(reader.fieldnames or [])
        if "frame" not in header:
            raise ValueError(
                f"{path}: not a TST motion evidence file (no 'frame' column)"
            )
        suffix = "_" + _STATE_SUFFIX
        mice = [name[:-len(suffix)] for name in header if name.endswith(suffix)]
        if not mice:
            raise ValueError(
                f"{path}: not a TST motion evidence file "
                f"(no '*_{_STATE_SUFFIX}' columns)"
            )
        evidence: Dict[str, List[Dict[str, object]]] = {
            mouse: [] for mouse in mice
        }
        for line, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame"])
                timestamp = _optional_float(row.get("timestamp_s", ""))
                for mouse in mice:
                    evidence[mouse].append(
                        _evidence_row(row, mouse, frame, timestamp)
                    )
            except ValueError as error:
                raise ValueError(f"{path}, line {line}: {error}") from None
    return evidence


def _evidence_row(
    row: Mapping[str, str], mouse: str, frame: int, timestamp: Optional[float]
) -> Dict[str, object]:
    def cell(name: str) -> str:
        return row.get(f"{mouse}_{name}") or ""

    item: Dict[str, object] = {
        "source_frame": frame,
        "timestamp_s": timestamp,
        "state": cell(_STATE_SUFFIX).strip() or "uncertain",
        "active_evidence": _optional_float(cell("active_evidence")),
    }
    immobile = cell("is_immobile").strip()
    item["is_immobile"] = bool(int(immobile)) if immobile else None
    for name in _EVIDENCE_FLOAT_COLUMNS:
        item[name] = float(cell(name) or 0.0)
    for name in _EVIDENCE_FLAG_COLUMNS:
        item[name] = bool(int(cell(name) or 0))
    item["qc_flag"] = cell("qc_flag")
    return item


# ---------------------------------------------------------------------------
# Immobility-time agreement (feeds core.agreement_stats.summarize_agreement)
# ---------------------------------------------------------------------------

@dataclass
class ImmobilityTimeAgreement:
    """Automated versus manual immobility time for one mouse.

    Both times cover the same frames: those annotated manually as something
    other than ``uncertain`` (and inside the scored window when the evidence
    carries ``is_immobile``). ``difference_s`` is ``auto - manual``.
    """
    mouse: str
    annotated_frames: int
    auto_s: float
    manual_s: float
    difference_s: float
    fps: float


def estimate_evidence_fps(evidence: Sequence[Mapping[str, object]]) -> float:
    """Frame rate implied by ``source_frame`` and ``timestamp_s`` (0 if unknown)."""
    points = [
        (float(item["source_frame"]), float(item["timestamp_s"]))
        for item in evidence
        if item.get("source_frame") is not None
        and item.get("timestamp_s") is not None
    ]
    if len(points) < 2:
        return 0.0
    frame_span = points[-1][0] - points[0][0]
    time_span = points[-1][1] - points[0][1]
    return frame_span / time_span if time_span > 0 else 0.0


def _auto_immobile(
    item: Mapping[str, object], smoothed: bool
) -> Optional[bool]:
    """Immobility of one evidence row; None outside the scored window."""
    if smoothed and "is_immobile" in item:
        flag = item["is_immobile"]
        return None if flag is None else bool(flag)
    return str(item.get("state", "uncertain")) in IMMOBILE_STATES


def auto_immobility_seconds(
    evidence: Sequence[Mapping[str, object]],
    fps: Optional[float] = None,
    smoothed: bool = True,
) -> float:
    """Total automated immobility seconds over a mouse's evidence rows.

    With ``smoothed=True`` (default) this uses the exported ``is_immobile``
    flag, the quantity behind ``total_immobility_time`` in the summary; rows
    outside the scored window (flag ``None``) do not count. Without that
    column, or with ``smoothed=False``, the per-frame ``state`` is used.
    ``fps`` defaults to the rate implied by the evidence.
    """
    rate = fps or estimate_evidence_fps(evidence)
    if not rate or rate <= 0:
        raise ValueError(
            "fps is required when the evidence has no frame/timestamp columns"
        )
    return sum(_auto_immobile(item, smoothed) is True for item in evidence) / rate


def immobility_time_agreement(
    motion_evidence: Mapping[str, Sequence[Mapping[str, object]]],
    annotations: Mapping[Tuple[str, int], str],
    fps: Optional[float] = None,
    smoothed: bool = True,
) -> Dict[str, ImmobilityTimeAgreement]:
    """Per-mouse immobility seconds, automated versus manual frame labels.

    ``annotations`` is the ``load_annotations`` mapping. Manual immobility is
    ``immobile_static`` or ``passive_swing``. Only frames with a manual label
    other than ``uncertain`` are compared, and the frame rate is the same for
    both sides, so annotate every frame of the scored window (expand bouts to
    frames) for the seconds to equal total immobility time. Mice without any
    comparable frame are omitted. Evidence rows without ``source_frame`` cannot
    be matched to annotations and are skipped.
    """
    results: Dict[str, ImmobilityTimeAgreement] = {}
    for mouse, evidence in motion_evidence.items():
        rows = list(evidence)
        rate = fps or estimate_evidence_fps(rows)
        if not rate or rate <= 0:
            raise ValueError(
                f"fps is required to convert frames to seconds for {mouse!r}"
            )
        frames = auto_frames = manual_frames = 0
        for item in rows:
            if item.get("source_frame") is None:
                continue
            truth = annotations.get((mouse, int(item["source_frame"])))
            automated = _auto_immobile(item, smoothed)
            if truth is None or truth == "uncertain" or automated is None:
                continue
            frames += 1
            auto_frames += automated
            manual_frames += truth in IMMOBILE_STATES
        if not frames:
            continue
        auto_s, manual_s = auto_frames / rate, manual_frames / rate
        results[mouse] = ImmobilityTimeAgreement(
            mouse=mouse, annotated_frames=frames, auto_s=auto_s,
            manual_s=manual_s, difference_s=auto_s - manual_s, fps=float(rate),
        )
    return results


def immobility_time_pairs(
    auto_seconds: Mapping[str, float],
    manual_seconds: Mapping[str, float],
) -> List[Tuple[float, float]]:
    """``(auto, manual)`` immobility seconds for keys present on both sides.

    Feed the result to ``core.agreement_stats.summarize_agreement``.
    ``auto_seconds`` can be ``TSTResult.total_immobility_time`` and
    ``manual_seconds`` the stopwatch totals, keyed identically (for example
    ``"<video>/<mouse>"`` across a batch). Order follows the sorted keys.
    """
    return [
        (auto_seconds[key], manual_seconds[key])
        for key in sorted(set(auto_seconds) & set(manual_seconds))
    ]


def agreement_pairs(
    agreements: Mapping[str, ImmobilityTimeAgreement],
) -> List[Tuple[float, float]]:
    """``(auto_s, manual_s)`` pairs from :func:`immobility_time_agreement`."""
    return [
        (agreements[mouse].auto_s, agreements[mouse].manual_s)
        for mouse in sorted(agreements)
    ]
