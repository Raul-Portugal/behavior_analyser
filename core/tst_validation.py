"""Transparent evaluation helpers for independently annotated TST frames.

This module deliberately does not create annotation files during normal runs.
It is for validating a fixed engine/profile against a separately prepared
ground-truth CSV when such data becomes available.
"""
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


VALID_STATES = {"immobile_static", "passive_swing", "active_escape", "uncertain"}


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
