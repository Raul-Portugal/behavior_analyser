"""Qt-free computation of the criterion-validation (agreement) report.

Inputs are the outputs of automated runs (``*_summary.json``, ``*_arm_entries.csv``,
``*_tst_motion_evidence.csv``, ``*_arena_zones.csv``) and manual annotation files
(``*_manual_arm_entries.csv`` and friends from the manual-scoring dialog, or
files of the same shape), matched per video by filename stem. Every number is
produced by the validation helpers in :mod:`core.positional_validation`,
:mod:`core.tst_validation` and :mod:`core.agreement_stats`; this module only
finds, pairs, pools and tabulates. Nothing here changes an analysis output.

:func:`build_results` returns a :class:`ValidationResults`;
:func:`build_tables` turns it into plain tables (the HTML report and the CSV
files are rendered from the same tables) and
:mod:`core.validation_report_render` writes the self-contained HTML.
"""
import csv
import hashlib
import json
import math
import operator
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.agreement_stats import (AgreementSummary, KappaResult, cohens_kappa,
                                  percent_agreement, summarize_agreement)
from core.manual_scoring import (CENTER, PERIPHERY, PARADIGM_EPM,
                                 PARADIGM_LABELS, PARADIGM_OPEN_FIELD,
                                 PARADIGM_TST, PARADIGM_Y_MAZE, TST_STATES,
                                 center_time_s, load_epm_totals,
                                 load_zone_intervals, zone_frame_labels)
from core.position_spotcheck import load_spotcheck_csv
from core.positional_validation import (EventMatchResult, compare_epm,
                                        compare_y_maze, epm_arm_class,
                                        epm_totals_from_entries,
                                        load_entry_annotations, match_events,
                                        normalize_epm_arm, position_errors)
from core.tst_validation import (IMMOBILE_STATES, evaluate_annotations,
                                 immobility_time_agreement, load_annotations,
                                 load_tst_evidence_csv)
from core.version import APP_VERSION

NAN = float("nan")
PARADIGM_ORDER = (PARADIGM_Y_MAZE, PARADIGM_EPM, PARADIGM_OPEN_FIELD,
                  PARADIGM_TST)

# Roles of the files a video can have.
AUTO_ROLES = ("summary", "arm_entries", "tracking", "tst_evidence",
              "arena_zones")
MANUAL_ROLES = ("manual_arm_entries", "manual_epm_totals",
                "manual_zone_intervals", "manual_tst_annotations",
                "spotcheck")
ALL_ROLES = AUTO_ROLES + MANUAL_ROLES

_AUTO_SUFFIXES = {
    "arm_entries": "_arm_entries.csv",
    "tracking": "_tracking_data.csv",
    "tst_evidence": "_tst_motion_evidence.csv",
    "arena_zones": "_arena_zones.csv",
}
_MANUAL_SUFFIXES = {
    "manual_arm_entries": "_manual_arm_entries.csv",
    "manual_epm_totals": "_manual_epm_totals.csv",
    "manual_zone_intervals": "_manual_zone_intervals.csv",
    "manual_tst_annotations": "_manual_tst_annotations.csv",
    "spotcheck": "_position_spotcheck.csv",
}
# CSVs written by the analysis that must never be mistaken for manual scores.
_AUTO_CSV_ENDINGS = tuple(_AUTO_SUFFIXES.values()) + (
    "_analysis_summary.csv", "_speed_data.csv", "_time_bins.csv",
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

@dataclass
class VideoInputs:
    """Everything known about one video; any file may be missing."""
    stem: str
    paradigm: Optional[str] = None
    summary: Optional[Path] = None
    arm_entries: Optional[Path] = None
    tracking: Optional[Path] = None
    tst_evidence: Optional[Path] = None
    arena_zones: Optional[Path] = None
    manual_arm_entries: Optional[Path] = None
    manual_epm_totals: Optional[Path] = None
    manual_zone_intervals: Optional[Path] = None
    manual_tst_annotations: Optional[Path] = None
    spotcheck: Optional[Path] = None

    def files(self) -> Dict[str, Path]:
        return {role: getattr(self, role) for role in ALL_ROLES
                if getattr(self, role) is not None}

    def merge(self, other: "VideoInputs") -> None:
        """Fill the roles ``self`` lacks from ``other``."""
        for role in ALL_ROLES:
            if getattr(self, role) is None and getattr(other, role) is not None:
                setattr(self, role, getattr(other, role))
        if self.paradigm is None:
            self.paradigm = other.paradigm


def detect_paradigm(summary: Mapping) -> Optional[str]:
    """The paradigm a ``*_summary.json`` was produced by."""
    if "tst_metrics" in summary:
        return PARADIGM_TST
    if "freestyle_metrics" in summary:
        return PARADIGM_OPEN_FIELD
    name = str((summary.get("run_configuration") or {}).get("paradigm_id") or "")
    lowered = name.lower()
    if "y-maze" in lowered or "y maze" in lowered:
        return PARADIGM_Y_MAZE
    if "plus" in lowered or "epm" in lowered:
        return PARADIGM_EPM
    if "freestyle" in lowered or "open field" in lowered:
        return PARADIGM_OPEN_FIELD
    if "tail" in lowered:
        return PARADIGM_TST
    specific = summary.get("specific_metrics") or {}
    if "arm_sequence" in specific:
        return PARADIGM_Y_MAZE
    if "time_in_open_arms_s" in specific:
        return PARADIGM_EPM
    return None


def _as_paths(sources) -> List[Path]:
    if isinstance(sources, (str, Path)):
        return [Path(sources)]
    return [Path(source) for source in sources]


def _iter_files(sources, pattern: str) -> Iterable[Path]:
    for source in _as_paths(sources):
        if source.is_file():
            yield source
        elif source.is_dir():
            yield from sorted(source.rglob(pattern))


def _unique_key(table: Mapping[str, object], stem: str) -> str:
    key, number = stem, 2
    while key in table:
        key = f"{stem} #{number}"
        number += 1
    return key


def discover_automated(sources) -> Dict[str, VideoInputs]:
    """Automated runs found in folders (searched recursively) or summary files.

    One entry per ``<base>_summary.json``; the sibling CSVs of the same base
    are attached. A stem that occurs in several runs gets ``#2``, ``#3``...
    suffixes so no run is silently dropped.
    """
    found: Dict[str, VideoInputs] = {}
    for path in _iter_files(sources, "*_summary.json"):
        name = path.name
        if not name.endswith("_summary.json") \
                or name.endswith("_position_spotcheck_summary.json"):
            continue
        base = name[:-len("_summary.json")]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        item = VideoInputs(
            stem=base,
            paradigm=detect_paradigm(data) if isinstance(data, dict) else None,
            summary=path,
        )
        for role, suffix in _AUTO_SUFFIXES.items():
            sibling = path.with_name(f"{base}{suffix}")
            if sibling.is_file():
                setattr(item, role, sibling)
        found[_unique_key(found, base)] = item
    return found


def classify_manual_file(path) -> Optional[Tuple[str, str]]:
    """``(role, stem)`` of a manual annotation CSV, or None if it is not one.

    Files named like the manual-scoring outputs are recognised by name; any
    other CSV is recognised by its header (so a scorer's own file with the
    right columns works, matched by its filename stem).
    """
    path = Path(path)
    name = path.name
    if path.suffix.lower() != ".csv":
        return None
    for role, suffix in _MANUAL_SUFFIXES.items():
        if name.endswith(suffix):
            return role, name[:-len(suffix)]
    if name.endswith(_AUTO_CSV_ENDINGS) or name.startswith("batch_"):
        return None
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    columns = {str(cell).strip().lower() for cell in header}
    stem = path.stem
    if {"mouse_name", "frame", "state"} <= columns:
        return "manual_tst_annotations", stem
    if {"zone", "start_frame", "end_frame"} <= columns:
        return "manual_zone_intervals", stem
    if {"manual_x", "auto_x", "frame"} <= columns:
        return "spotcheck", stem
    if {"metric", "value"} <= columns:
        return "manual_epm_totals", stem
    if "arm" in columns and columns & {"timestamp_s", "timestamp", "time_s", "time"}:
        return "manual_arm_entries", stem
    return None


def discover_manual(sources) -> Dict[str, VideoInputs]:
    """Manual annotation files found in folders or given directly, by stem."""
    found: Dict[str, VideoInputs] = {}
    for path in _iter_files(sources, "*.csv"):
        classified = classify_manual_file(path)
        if classified is None:
            continue
        role, stem = classified
        item = found.get(stem)
        if item is None:
            item = found[stem] = VideoInputs(stem=stem)
        if getattr(item, role) is None:
            setattr(item, role, path)
    return found


def combine_inputs(
    automated: Mapping[str, VideoInputs],
    manual: Mapping[str, VideoInputs],
) -> List[VideoInputs]:
    """Pair automated and manual files by filename stem.

    Stems are compared case-insensitively. Manual files without an automated
    run (and the reverse) stay in the list so the caller can review and fix
    the matching. A stem with duplicate runs (``name #2``) matches the manual
    files of ``name`` for its first run only.
    """
    combined: Dict[str, VideoInputs] = {}
    lower: Dict[str, str] = {}
    for stem, item in automated.items():
        copy = VideoInputs(stem=stem)
        copy.merge(item)
        combined[stem] = copy
        lower.setdefault(Path(stem.split(" #")[0]).name.lower(), stem)
    for stem, item in manual.items():
        key = lower.get(stem.lower())
        if key is None or key not in combined:
            key = _unique_key(combined, stem)
            combined[key] = VideoInputs(stem=stem)
        combined[key].merge(item)
    for item in combined.values():
        if item.paradigm is None:
            item.paradigm = infer_paradigm_from_files(item)
    return sorted(combined.values(), key=lambda v: v.stem.lower())


def infer_paradigm_from_files(item: VideoInputs) -> Optional[str]:
    if item.manual_tst_annotations or item.tst_evidence:
        return PARADIGM_TST
    if item.manual_zone_intervals or item.arena_zones:
        return PARADIGM_OPEN_FIELD
    if item.manual_epm_totals:
        return PARADIGM_EPM
    return None


# --------------------------------------------------------------------------
# Configuration, criteria
# --------------------------------------------------------------------------

_OPERATORS = {">=": operator.ge, "<=": operator.le, ">": operator.gt,
              "<": operator.lt}


@dataclass
class Criterion:
    """``statistic <op> threshold``; ``key`` is ``paradigm.metric.statistic``."""
    key: str
    op: str
    threshold: float

    def __post_init__(self):
        if self.op not in _OPERATORS:
            raise ValueError(f"Unknown comparison {self.op!r}")
        self.threshold = float(self.threshold)

    def check(self, value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "not evaluable"
        return "pass" if _OPERATORS[self.op](value, self.threshold) else "fail"


@dataclass
class CriterionResult:
    criterion: Criterion
    label: str
    value: Optional[float]
    status: str          # "pass" | "fail" | "not evaluable"


@dataclass
class ReportConfig:
    title: str = "Criterion validation report"
    notes: str = ""
    tolerance_s: float = 1.0
    confidence: float = 0.95
    criteria: List[Criterion] = field(default_factory=list)


# Pooled-agreement metrics per paradigm: key -> (label, unit).
METRICS: Dict[str, Dict[str, Tuple[str, str]]] = {
    PARADIGM_Y_MAZE: {
        "alternation_pct": ("Alternation", "%"),
        "total_entries": ("Total arm entries", "entries"),
    },
    PARADIGM_EPM: {
        "percent_time_in_open": ("Time in open arms", "%"),
        "percent_entries_to_open": ("Entries to open arms", "%"),
        "total_arm_entries": ("Total arm entries", "entries"),
        "entries_to_open_arms": ("Open-arm entries", "entries"),
        "time_in_open_arms_s": ("Time in open arms", "s"),
    },
    PARADIGM_OPEN_FIELD: {
        "center_time_s": ("Center time", "s"),
    },
    PARADIGM_TST: {
        "immobility_s": ("Immobility time", "s"),
    },
}

AGREEMENT_STATS: Dict[str, str] = {
    "icc": "ICC(2,1)",
    "icc_ci_low": "ICC(2,1) lower CI bound",
    "icc_ci_high": "ICC(2,1) upper CI bound",
    "bias": "Bland-Altman bias (auto - manual)",
    "abs_bias": "|Bland-Altman bias|",
    "loa_half_width": "Limits of agreement half-width",
    "slope_p": "Proportional-bias p-value",
    "pearson_r": "Pearson r",
    "spearman_rho": "Spearman rho",
    "rmse": "RMSE",
    "mae": "MAE",
    "mapd": "Mean absolute % difference",
}
EVENT_STATS = {"precision": "Event precision", "recall": "Event recall",
               "f1": "Event F1"}
FRAME_STATS = {
    PARADIGM_TST: {
        "kappa": "State kappa", "percent_agreement": "State agreement (%)",
        "immobility_precision": "Immobility precision",
        "immobility_recall": "Immobility recall",
        "immobility_f1": "Immobility F1",
    },
    PARADIGM_OPEN_FIELD: {
        "kappa": "Zone kappa", "percent_agreement": "Zone agreement (%)",
    },
}
POSITION_STATS = {
    "miss_rate": "Miss rate", "false_detection_rate": "False-detection rate",
    "median_error_px": "Median error (px)", "p95_error_px": "95th percentile error (px)",
    "median_error_cm": "Median error (cm)", "p95_error_cm": "95th percentile error (cm)",
}


def criteria_catalog() -> List[Tuple[str, str]]:
    """Every criterion key with a readable label, for choosing in the GUI."""
    entries: List[Tuple[str, str]] = []
    for paradigm in PARADIGM_ORDER:
        label = PARADIGM_LABELS[paradigm]
        for metric, (metric_label, unit) in METRICS[paradigm].items():
            for stat, stat_label in AGREEMENT_STATS.items():
                entries.append((f"{paradigm}.{metric}.{stat}",
                                f"{label}: {metric_label} ({unit}) - {stat_label}"))
        if paradigm in (PARADIGM_Y_MAZE, PARADIGM_EPM):
            for stat, stat_label in EVENT_STATS.items():
                entries.append((f"{paradigm}.events.{stat}",
                                f"{label}: {stat_label}"))
        for stat, stat_label in FRAME_STATS.get(paradigm, {}).items():
            entries.append((f"{paradigm}.frames.{stat}",
                            f"{label}: {stat_label}"))
    for stat, stat_label in POSITION_STATS.items():
        entries.append((f"position.spotcheck.{stat}",
                        f"Position spot-check: {stat_label}"))
    return entries


@lru_cache(maxsize=1)
def _catalog_labels() -> Dict[str, str]:
    return dict(criteria_catalog())


def criterion_label(key: str) -> str:
    return _catalog_labels().get(key, key)


def default_criteria() -> List[Criterion]:
    """The pre-registration defaults of ``VALIDATION_PROTOCOL.md`` section 5."""
    criteria = [
        Criterion("y_maze.alternation_pct.icc", ">=", 0.90),
        Criterion("y_maze.alternation_pct.icc_ci_low", ">=", 0.75),
        Criterion("y_maze.alternation_pct.abs_bias", "<=", 5.0),
        Criterion("y_maze.alternation_pct.slope_p", ">=", 0.05),
        Criterion("y_maze.events.f1", ">=", 0.90),
        Criterion("y_maze.events.precision", ">=", 0.85),
        Criterion("y_maze.events.recall", ">=", 0.85),
        Criterion("epm.percent_time_in_open.icc", ">=", 0.90),
        Criterion("epm.percent_time_in_open.icc_ci_low", ">=", 0.75),
        Criterion("epm.percent_time_in_open.abs_bias", "<=", 3.0),
        Criterion("epm.events.f1", ">=", 0.90),
        Criterion("open_field.center_time_s.icc", ">=", 0.90),
        Criterion("open_field.center_time_s.icc_ci_low", ">=", 0.75),
        Criterion("open_field.frames.kappa", ">=", 0.80),
        Criterion("tst.immobility_s.icc", ">=", 0.90),
        Criterion("tst.immobility_s.icc_ci_low", ">=", 0.75),
        Criterion("tst.immobility_s.abs_bias", "<=", 5.0),
        Criterion("tst.frames.kappa", ">=", 0.80),
        Criterion("tst.frames.immobility_precision", ">=", 0.85),
        Criterion("tst.frames.immobility_recall", ">=", 0.85),
    ]
    return criteria


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------

@dataclass
class MetricAgreement:
    """One pooled endpoint: the per-video pairs and their agreement summary."""
    paradigm: str
    key: str
    label: str
    unit: str
    pairs: List[Tuple[float, float]]
    pair_labels: List[str]
    summary: AgreementSummary


@dataclass
class PooledEvents:
    """Event matching pooled over videos (counts summed, then rates)."""
    paradigm: str
    tolerance_s: float
    n_videos: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    macro_f1: float
    mean_offset_s: float
    mean_abs_offset_s: float


@dataclass
class ConfusionResult:
    """Pooled frame labels: rows automated, columns manual."""
    name: str
    title: str
    categories: List[str]
    matrix: np.ndarray
    kappa: KappaResult
    percent_agreement: float
    n_videos: int


@dataclass
class SpotcheckPooled:
    n_files: int
    n_reviewed: int
    n_visible: int
    n_not_visible: int
    n_matched: int
    n_missed: int
    miss_rate: float
    false_detections: int
    false_detection_rate: float
    mean_px: float
    median_px: float
    p95_px: float
    mean_cm: float
    median_cm: float
    p95_cm: float
    errors_px: np.ndarray
    errors_cm: np.ndarray


@dataclass
class ParadigmResult:
    paradigm: str
    columns: List[str] = field(default_factory=list)
    rows: List[List[object]] = field(default_factory=list)
    agreements: Dict[str, MetricAgreement] = field(default_factory=dict)
    events: Optional[PooledEvents] = None
    confusions: List[ConfusionResult] = field(default_factory=list)
    frame_stats: Dict[str, float] = field(default_factory=dict)
    n_videos: int = 0


@dataclass
class InputFile:
    role: str
    stem: str
    path: str
    size_bytes: int
    sha256: str


@dataclass
class ValidationResults:
    config: ReportConfig
    paradigms: Dict[str, ParadigmResult]
    spotcheck_columns: List[str]
    spotcheck_rows: List[List[object]]
    spotcheck: Optional[SpotcheckPooled]
    criteria: List[CriterionResult]
    inputs: List[VideoInputs]
    input_files: List[InputFile]
    warnings: List[str]
    generated: str
    app_version: str = APP_VERSION


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _num(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return NAN
    return value


def _agreement(
    paradigm: str, key: str, pairs: Sequence[Tuple[float, float]],
    labels: Sequence[str], confidence: float,
) -> MetricAgreement:
    label, unit = METRICS[paradigm][key]
    return MetricAgreement(
        paradigm, key, label, unit, [(a, m) for a, m in pairs], list(labels),
        summarize_agreement(pairs, confidence),
    )


def pool_events(
    paradigm: str, matchings: Sequence[EventMatchResult], tolerance_s: float
) -> Optional[PooledEvents]:
    """Micro-pooled precision/recall/F1 plus the mean of per-video F1."""
    if not matchings:
        return None
    tp = sum(m.true_positives for m in matchings)
    fp = sum(m.false_positives for m in matchings)
    fn = sum(m.false_negatives for m in matchings)
    offsets = np.array([offset for m in matchings for _, _, offset in m.matches],
                       dtype=float)
    f1s = [m.f1 for m in matchings if math.isfinite(m.f1)]
    return PooledEvents(
        paradigm=paradigm, tolerance_s=float(tolerance_s),
        n_videos=len(matchings), true_positives=tp, false_positives=fp,
        false_negatives=fn,
        precision=tp / (tp + fp) if tp + fp else NAN,
        recall=tp / (tp + fn) if tp + fn else NAN,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else NAN,
        macro_f1=float(np.mean(f1s)) if f1s else NAN,
        mean_offset_s=float(offsets.mean()) if offsets.size else NAN,
        mean_abs_offset_s=float(np.abs(offsets).mean()) if offsets.size else NAN,
    )


def _load_summary(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _auto_entries(item: VideoInputs, maze: str):
    """Automated ``(arm, time)`` entries from the CSV, else the summary JSON."""
    if item.arm_entries is not None:
        return load_entry_annotations(item.arm_entries, maze)
    summary = _load_summary(item.summary)
    entries = (summary.get("specific_metrics") or {}).get("arm_entries")
    if entries is None:
        return None
    return [(str(arm), float(stamp)) for arm, stamp in entries]


def _confusion(
    name: str, title: str, pairs: Sequence[Tuple[str, str]],
    categories: Sequence[str], n_videos: int,
) -> ConfusionResult:
    extra = sorted({label for pair in pairs for label in pair} - set(categories))
    categories = list(categories) + extra
    kappa = cohens_kappa([a for a, _ in pairs], [m for _, m in pairs],
                         categories=categories)
    return ConfusionResult(
        name, title, categories, kappa.confusion, kappa,
        percent_agreement([a for a, _ in pairs], [m for _, m in pairs])
        if pairs else NAN,
        n_videos,
    )


# --------------------------------------------------------------------------
# Paradigm evaluations. Each takes the videos of its paradigm and returns a
# ParadigmResult; a video that cannot be evaluated adds a warning instead.
# --------------------------------------------------------------------------

def _evaluate_y_maze(videos, config, warnings) -> ParadigmResult:
    result = ParadigmResult(PARADIGM_Y_MAZE)
    result.columns = [
        "video", "auto_entries", "manual_entries", "entries_difference",
        "auto_alternation_pct", "manual_alternation_pct",
        "alternation_difference_pct", "true_positives", "false_positives",
        "false_negatives", "precision", "recall", "f1", "mean_abs_offset_s",
        "sequence_similarity", "auto_sequence", "manual_sequence",
    ]
    alternation, totals, matchings, labels = [], [], [], []
    for item in videos:
        if item.manual_arm_entries is None:
            warnings.append(f"{item.stem}: no manual arm entries (Y-maze)")
            continue
        try:
            auto = _auto_entries(item, "y_maze")
            if auto is None:
                warnings.append(f"{item.stem}: no automated arm entries")
                continue
            manual = load_entry_annotations(item.manual_arm_entries, "y_maze")
            comparison = compare_y_maze(auto, manual, config.tolerance_s)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{item.stem}: {error}")
            continue
        match = comparison.matching
        result.rows.append([
            item.stem, comparison.auto_total_entries,
            comparison.manual_total_entries, comparison.total_entries_difference,
            comparison.auto_alternation_pct, comparison.manual_alternation_pct,
            comparison.alternation_difference_pct, match.true_positives,
            match.false_positives, match.false_negatives, match.precision,
            match.recall, match.f1, match.mean_abs_offset_s,
            comparison.edit_distance.similarity, comparison.auto_sequence,
            comparison.manual_sequence,
        ])
        alternation.append((comparison.auto_alternation_pct,
                            comparison.manual_alternation_pct))
        totals.append((comparison.auto_total_entries,
                       comparison.manual_total_entries))
        matchings.append(match)
        labels.append(item.stem)
    result.n_videos = len(result.rows)
    if result.rows:
        result.agreements["alternation_pct"] = _agreement(
            PARADIGM_Y_MAZE, "alternation_pct", alternation, labels,
            config.confidence)
        result.agreements["total_entries"] = _agreement(
            PARADIGM_Y_MAZE, "total_entries", totals, labels,
            config.confidence)
        result.events = pool_events(PARADIGM_Y_MAZE, matchings,
                                    config.tolerance_s)
    return result


EPM_COLUMN_METRICS = (
    "percent_time_in_open", "percent_entries_to_open", "total_arm_entries",
    "entries_to_open_arms", "time_in_open_arms_s",
)


def _normalized_epm(entries):
    return [(normalize_epm_arm(arm), float(stamp)) for arm, stamp in entries]


def _evaluate_epm(videos, config, warnings) -> ParadigmResult:
    result = ParadigmResult(PARADIGM_EPM)
    columns = ["video"]
    for metric in EPM_COLUMN_METRICS:
        columns += [f"auto_{metric}", f"manual_{metric}", f"difference_{metric}"]
    columns += ["event_true_positives", "event_false_positives",
                "event_false_negatives", "event_precision", "event_recall",
                "event_f1"]
    result.columns = columns
    pairs: Dict[str, List[Tuple[float, float]]] = {
        metric: [] for metric in EPM_COLUMN_METRICS}
    pair_labels: Dict[str, List[str]] = {m: [] for m in EPM_COLUMN_METRICS}
    matchings = []
    for item in videos:
        if item.manual_arm_entries is None and item.manual_epm_totals is None:
            warnings.append(f"{item.stem}: no manual EPM scoring")
            continue
        if item.summary is None:
            warnings.append(f"{item.stem}: no automated summary (EPM)")
            continue
        try:
            auto_fields = (_load_summary(item.summary).get("specific_metrics")
                           or {})
            manual_entries = None
            manual_totals: Dict[str, float] = {}
            if item.manual_arm_entries is not None:
                manual_entries = load_entry_annotations(
                    item.manual_arm_entries, "epm")
                manual_totals.update(epm_totals_from_entries(manual_entries))
            if item.manual_epm_totals is not None:
                manual_totals.update(load_epm_totals(item.manual_epm_totals))
            differences = compare_epm(auto_fields, manual_totals)
            auto_entries = (
                _auto_entries(item, "epm") if manual_entries is not None
                else None)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{item.stem}: {error}")
            continue
        if not differences:
            warnings.append(f"{item.stem}: no EPM metric present on both sides")
            continue
        row: List[object] = [item.stem]
        for metric in EPM_COLUMN_METRICS:
            diff = differences.get(metric)
            if diff is None:
                row += [None, None, None]
                continue
            row += [diff.auto, diff.manual, diff.difference]
            pairs[metric].append((diff.auto, diff.manual))
            pair_labels[metric].append(item.stem)
        event_cells: List[object] = [None] * 6
        if auto_entries is not None and manual_entries is not None:
            auto_n, manual_n = (_normalized_epm(auto_entries),
                                _normalized_epm(manual_entries))
            coarse = any(arm in ("open", "closed") for arm, _ in manual_n)
            match = match_events(
                auto_n, manual_n, config.tolerance_s,
                arm_key=epm_arm_class if coarse else None)
            matchings.append(match)
            event_cells = [match.true_positives, match.false_positives,
                           match.false_negatives, match.precision,
                           match.recall, match.f1]
        result.rows.append(row + event_cells)
    result.n_videos = len(result.rows)
    for metric in EPM_COLUMN_METRICS:
        if pairs[metric]:
            result.agreements[metric] = _agreement(
                PARADIGM_EPM, metric, pairs[metric], pair_labels[metric],
                config.confidence)
    result.events = pool_events(PARADIGM_EPM, matchings, config.tolerance_s)
    return result


def _read_arena_labels(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            labels[int(row["frame"])] = str(row["arena_zone"]).strip().lower()
    return labels


def _evaluate_open_field(videos, config, warnings) -> ParadigmResult:
    result = ParadigmResult(PARADIGM_OPEN_FIELD)
    result.columns = [
        "video", "auto_center_time_s", "manual_center_time_s",
        "difference_s", "manual_scored_s", "frames_compared",
        "frame_kappa", "frame_percent_agreement",
    ]
    pairs, labels = [], []
    frame_pairs: List[Tuple[str, str]] = []
    n_label_videos = 0
    for item in videos:
        if item.manual_zone_intervals is None:
            warnings.append(f"{item.stem}: no manual zone intervals")
            continue
        if item.summary is None:
            warnings.append(f"{item.stem}: no automated summary (open field)")
            continue
        try:
            arena = ((_load_summary(item.summary).get("freestyle_metrics") or {})
                     .get("arena"))
            intervals = load_zone_intervals(item.manual_zone_intervals)
            manual_labels = zone_frame_labels(intervals)
            auto_labels = (_read_arena_labels(item.arena_zones)
                           if item.arena_zones is not None else None)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{item.stem}: {error}")
            continue
        auto_center = _num((arena or {}).get("center_time_s"))
        if arena is None or not math.isfinite(auto_center):
            warnings.append(
                f"{item.stem}: the run has no open-field arena metrics "
                "(center time needs the arena option)")
            continue
        manual_center = center_time_s(intervals)
        kept: List[Tuple[str, str]] = []
        if auto_labels is not None:
            for frame in sorted(set(auto_labels) & set(manual_labels)):
                auto_label = auto_labels[frame]
                if auto_label in (CENTER, PERIPHERY):
                    kept.append((auto_label, manual_labels[frame]))
        kappa = NAN
        agreement = NAN
        if kept:
            frame_pairs.extend(kept)
            n_label_videos += 1
            one = cohens_kappa([a for a, _ in kept], [m for _, m in kept],
                               categories=[CENTER, PERIPHERY])
            kappa = one.kappa
            agreement = one.observed_agreement * 100.0
        result.rows.append([
            item.stem, auto_center, manual_center, auto_center - manual_center,
            sum(i.duration_s for i in intervals), len(kept), kappa, agreement,
        ])
        pairs.append((auto_center, manual_center))
        labels.append(item.stem)
    result.n_videos = len(result.rows)
    if result.rows:
        result.agreements["center_time_s"] = _agreement(
            PARADIGM_OPEN_FIELD, "center_time_s", pairs, labels,
            config.confidence)
    if frame_pairs:
        confusion = _confusion(
            "open_field_zones", "Open-field zone (frame level)", frame_pairs,
            [CENTER, PERIPHERY], n_label_videos)
        result.confusions.append(confusion)
        result.frame_stats = {
            "kappa": confusion.kappa.kappa,
            "percent_agreement": confusion.percent_agreement,
        }
    return result


def _evaluate_tst(videos, config, warnings) -> ParadigmResult:
    result = ParadigmResult(PARADIGM_TST)
    result.columns = [
        "video", "mouse", "annotated_frames", "exact_state_agreement_pct",
        "immobility_precision", "immobility_recall", "immobility_f1",
        "compared_frames", "auto_immobility_s", "manual_immobility_s",
        "difference_s",
    ]
    pairs, labels = [], []
    state_pairs: List[Tuple[str, str]] = []
    n_state_videos = 0
    for item in videos:
        if item.manual_tst_annotations is None:
            warnings.append(f"{item.stem}: no manual TST annotations")
            continue
        if item.tst_evidence is None:
            warnings.append(f"{item.stem}: no automated TST motion evidence")
            continue
        try:
            evidence = load_tst_evidence_csv(item.tst_evidence)
            annotations = load_annotations(item.manual_tst_annotations)
            per_mouse = evaluate_annotations(evidence, annotations)
            times = immobility_time_agreement(evidence, annotations)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{item.stem}: {error}")
            continue
        annotated_mice = {mouse for mouse, _ in annotations}
        for mouse in sorted(annotated_mice - set(evidence)):
            warnings.append(
                f"{item.stem}: annotations for {mouse!r}, which is not in the "
                "automated evidence")
        video_pairs: List[Tuple[str, str]] = []
        for mouse, rows in evidence.items():
            for row in rows:
                frame = row.get("source_frame")
                truth = annotations.get((mouse, int(frame))) \
                    if frame is not None else None
                if truth is not None:
                    video_pairs.append((str(row.get("state", "uncertain")),
                                        truth))
        if video_pairs:
            state_pairs.extend(video_pairs)
            n_state_videos += 1
        for mouse in sorted(per_mouse):
            stats = per_mouse[mouse]
            time = times.get(mouse)
            result.rows.append([
                item.stem, mouse, int(stats["annotated_frames"]),
                stats["exact_state_agreement_pct"],
                stats["immobility_precision"], stats["immobility_recall"],
                stats["immobility_f1"],
                time.annotated_frames if time else None,
                time.auto_s if time else None,
                time.manual_s if time else None,
                time.difference_s if time else None,
            ])
            if time is not None:
                pairs.append((time.auto_s, time.manual_s))
                labels.append(f"{item.stem}/{mouse}")
    result.n_videos = len({row[0] for row in result.rows})
    if pairs:
        result.agreements["immobility_s"] = _agreement(
            PARADIGM_TST, "immobility_s", pairs, labels, config.confidence)
    if state_pairs:
        confusion = _confusion(
            "tst_states", "TST behavior state (frame level)", state_pairs,
            list(TST_STATES), n_state_videos)
        result.confusions.append(confusion)
        immobile = IMMOBILE_STATES
        tp = sum(a in immobile and m in immobile for a, m in state_pairs)
        fp = sum(a in immobile and m not in immobile for a, m in state_pairs)
        fn = sum(a not in immobile and m in immobile for a, m in state_pairs)
        precision = tp / (tp + fp) if tp + fp else NAN
        recall = tp / (tp + fn) if tp + fn else NAN
        result.frame_stats = {
            "kappa": confusion.kappa.kappa,
            "percent_agreement": confusion.percent_agreement,
            "immobility_precision": precision,
            "immobility_recall": recall,
            "immobility_f1": (2 * tp / (2 * tp + fp + fn)
                              if 2 * tp + fp + fn else NAN),
        }
    return result


def pool_spotchecks(
    loaded: Sequence[Tuple[str, object]]
) -> Tuple[List[str], List[List[object]], Optional[SpotcheckPooled]]:
    """Per-file rows and the pooled position statistics of spot-check files."""
    columns = [
        "video", "reviewed", "visible", "matched", "missed", "miss_rate",
        "false_detections", "not_visible", "median_error_px", "p95_error_px",
        "median_error_cm", "scale_factor",
    ]
    rows: List[List[object]] = []
    all_px, all_cm = [], []
    totals = Counter()
    for stem, data in loaded:
        errors = position_errors(data.auto_xy, data.manual_xy,
                                 data.scale_factor)
        rows.append([
            stem, len(data.auto_xy), errors.n_manual, errors.n_matched,
            errors.n_missed, errors.miss_rate, errors.false_detections,
            len(data.auto_xy) - errors.n_manual, errors.median_px,
            errors.p95_px, errors.median_cm, data.scale_factor,
        ])
        all_px.append(errors.errors_px)
        all_cm.append(errors.errors_cm)
        totals["reviewed"] += len(data.auto_xy)
        totals["visible"] += errors.n_manual
        totals["matched"] += errors.n_matched
        totals["missed"] += errors.n_missed
        totals["false"] += errors.false_detections
    if not rows:
        return columns, rows, None
    px = np.concatenate(all_px) if all_px else np.array([])
    cm = np.concatenate(all_cm) if all_cm else np.array([])
    cm = cm[~np.isnan(cm)]

    def stats(values):
        if not values.size:
            return NAN, NAN, NAN
        return (float(values.mean()), float(np.median(values)),
                float(np.percentile(values, 95)))

    not_visible = totals["reviewed"] - totals["visible"]
    mean_px, median_px, p95_px = stats(px)
    mean_cm, median_cm, p95_cm = stats(cm)
    pooled = SpotcheckPooled(
        n_files=len(rows), n_reviewed=totals["reviewed"],
        n_visible=totals["visible"], n_not_visible=not_visible,
        n_matched=totals["matched"], n_missed=totals["missed"],
        miss_rate=(totals["missed"] / totals["visible"]
                   if totals["visible"] else NAN),
        false_detections=totals["false"],
        false_detection_rate=(totals["false"] / not_visible
                              if not_visible else NAN),
        mean_px=mean_px, median_px=median_px, p95_px=p95_px,
        mean_cm=mean_cm, median_cm=median_cm, p95_cm=p95_cm,
        errors_px=px, errors_cm=cm,
    )
    return columns, rows, pooled


# --------------------------------------------------------------------------
# Statistics for the criteria
# --------------------------------------------------------------------------

def agreement_statistics(summary: AgreementSummary) -> Dict[str, float]:
    """The ``AGREEMENT_STATS`` values of one pooled endpoint."""
    icc = summary.icc
    ba = summary.bland_altman
    return {
        "icc": icc.icc if icc else NAN,
        "icc_ci_low": icc.ci_low if icc else NAN,
        "icc_ci_high": icc.ci_high if icc else NAN,
        "bias": ba.bias,
        "abs_bias": abs(ba.bias) if math.isfinite(ba.bias) else NAN,
        "loa_half_width": (ba.loa_upper - ba.loa_lower) / 2.0,
        "slope_p": ba.slope_p_value,
        "pearson_r": summary.pearson.r,
        "spearman_rho": summary.spearman.r,
        "rmse": summary.rmse,
        "mae": summary.mae,
        "mapd": summary.mapd,
    }


def flat_statistics(
    paradigms: Mapping[str, ParadigmResult],
    spotcheck: Optional[SpotcheckPooled],
) -> Dict[str, float]:
    """Every computed statistic by ``paradigm.metric.statistic`` key."""
    values: Dict[str, float] = {}
    for paradigm, result in paradigms.items():
        for metric, agreement in result.agreements.items():
            for stat, value in agreement_statistics(agreement.summary).items():
                values[f"{paradigm}.{metric}.{stat}"] = float(value)
            values[f"{paradigm}.{metric}.n"] = float(agreement.summary.n)
        if result.events is not None:
            for stat in EVENT_STATS:
                values[f"{paradigm}.events.{stat}"] = float(
                    getattr(result.events, stat))
        for stat, value in result.frame_stats.items():
            values[f"{paradigm}.frames.{stat}"] = float(value)
    if spotcheck is not None:
        values.update({
            "position.spotcheck.miss_rate": spotcheck.miss_rate,
            "position.spotcheck.false_detection_rate":
                spotcheck.false_detection_rate,
            "position.spotcheck.median_error_px": spotcheck.median_px,
            "position.spotcheck.p95_error_px": spotcheck.p95_px,
            "position.spotcheck.median_error_cm": spotcheck.median_cm,
            "position.spotcheck.p95_error_cm": spotcheck.p95_cm,
        })
    return values


def evaluate_criteria(
    criteria: Sequence[Criterion], statistics: Mapping[str, float]
) -> List[CriterionResult]:
    """Mark each criterion pass, fail or not evaluable."""
    results = []
    for criterion in criteria:
        value = statistics.get(criterion.key)
        results.append(CriterionResult(
            criterion, criterion_label(criterion.key), value,
            criterion.check(value)))
    return results


# --------------------------------------------------------------------------
# Whole report
# --------------------------------------------------------------------------

def _hash_file(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def inventory_files(inputs: Sequence[VideoInputs],
                    warnings: List[str]) -> List[InputFile]:
    """Every input file with its size and SHA-256 (for reproducibility)."""
    files: List[InputFile] = []
    for item in inputs:
        for role, path in item.files().items():
            try:
                size, digest = _hash_file(path)
            except OSError as error:
                warnings.append(f"{item.stem}: cannot read {path.name}: {error}")
                continue
            files.append(InputFile(role, item.stem, str(path), size, digest))
    return files


def build_results(
    inputs: Sequence[VideoInputs], config: Optional[ReportConfig] = None,
) -> ValidationResults:
    """Evaluate every paradigm for the (already matched) inputs."""
    config = config or ReportConfig(criteria=default_criteria())
    warnings: List[str] = []
    by_paradigm: Dict[str, List[VideoInputs]] = {p: [] for p in PARADIGM_ORDER}
    for item in inputs:
        if item.paradigm in by_paradigm:
            by_paradigm[item.paradigm].append(item)
        elif any(getattr(item, role) is not None for role in MANUAL_ROLES
                 if role != "spotcheck"):
            warnings.append(
                f"{item.stem}: paradigm unknown (no automated run); set it "
                "in the matching table")
    evaluators = {
        PARADIGM_Y_MAZE: _evaluate_y_maze, PARADIGM_EPM: _evaluate_epm,
        PARADIGM_OPEN_FIELD: _evaluate_open_field, PARADIGM_TST: _evaluate_tst,
    }
    paradigms: Dict[str, ParadigmResult] = {}
    for paradigm in PARADIGM_ORDER:
        videos = by_paradigm[paradigm]
        if videos:
            result = evaluators[paradigm](videos, config, warnings)
            if result.rows:
                paradigms[paradigm] = result

    loaded = []
    for item in inputs:
        if item.spotcheck is not None:
            try:
                loaded.append((item.stem, load_spotcheck_csv(item.spotcheck)))
            except (OSError, ValueError) as error:
                warnings.append(f"{item.stem}: {error}")
    spot_columns, spot_rows, spot_pooled = pool_spotchecks(loaded)

    statistics = flat_statistics(paradigms, spot_pooled)
    return ValidationResults(
        config=config, paradigms=paradigms, spotcheck_columns=spot_columns,
        spotcheck_rows=spot_rows, spotcheck=spot_pooled,
        criteria=evaluate_criteria(config.criteria, statistics),
        inputs=list(inputs), input_files=inventory_files(inputs, warnings),
        warnings=warnings,
        generated=datetime.now().isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------
# Tables (rendered to HTML and written as CSV)
# --------------------------------------------------------------------------

@dataclass
class Table:
    name: str
    title: str
    columns: List[str]
    rows: List[List[object]]
    note: str = ""


def build_tables(results: ValidationResults) -> List[Table]:
    """All report tables, in report order."""
    tables: List[Table] = []
    tables.append(Table(
        "inputs", "Matched inputs",
        ["video", "paradigm"] + list(ALL_ROLES),
        [[item.stem, item.paradigm or ""]
         + [str(getattr(item, role)) if getattr(item, role) else ""
            for role in ALL_ROLES] for item in results.inputs]))

    table_names = {
        PARADIGM_Y_MAZE: ("y_maze_per_video", "Y-maze: per video"),
        PARADIGM_EPM: ("epm_per_video", "Elevated plus maze: per video"),
        PARADIGM_OPEN_FIELD: ("open_field_per_video",
                              "Open field: per video"),
        PARADIGM_TST: ("tst_per_mouse", "Tail suspension test: per mouse"),
    }
    for paradigm in PARADIGM_ORDER:
        result = results.paradigms.get(paradigm)
        if result is not None:
            name, title = table_names[paradigm]
            tables.append(Table(name, title, result.columns, result.rows))

    agreement_rows: List[List[object]] = []
    for paradigm in PARADIGM_ORDER:
        result = results.paradigms.get(paradigm)
        if result is None:
            continue
        for metric, agreement in result.agreements.items():
            s = agreement.summary
            icc, ba = s.icc, s.bland_altman
            agreement_rows.append([
                PARADIGM_LABELS[paradigm], agreement.label, agreement.unit,
                s.n, s.n_excluded, s.mean_auto, s.mean_manual,
                icc.icc if icc else None, icc.ci_low if icc else None,
                icc.ci_high if icc else None,
                ba.bias, ba.bias_ci[0], ba.bias_ci[1], ba.sd_diff,
                ba.loa_lower, ba.loa_upper, ba.slope, ba.slope_p_value,
                s.pearson.r, s.spearman.r, s.rmse, s.mae, s.mapd,
            ])
    tables.append(Table(
        "pooled_agreement", "Pooled agreement (automated vs manual)",
        ["paradigm", "metric", "unit", "n", "n_excluded", "mean_auto",
         "mean_manual", "icc_2_1", "icc_ci_low", "icc_ci_high", "bias",
         "bias_ci_low", "bias_ci_high", "sd_diff", "loa_lower", "loa_upper",
         "slope", "slope_p", "pearson_r", "spearman_rho", "rmse", "mae",
         "mapd"],
        agreement_rows,
        note="Differences are automated minus manual. n_excluded counts "
             "videos with a missing value on either side."))

    event_rows = []
    for paradigm in PARADIGM_ORDER:
        result = results.paradigms.get(paradigm)
        if result is not None and result.events is not None:
            e = result.events
            event_rows.append([
                PARADIGM_LABELS[paradigm], e.tolerance_s, e.n_videos,
                e.true_positives, e.false_positives, e.false_negatives,
                e.precision, e.recall, e.f1, e.macro_f1, e.mean_offset_s,
                e.mean_abs_offset_s])
    if event_rows:
        tables.append(Table(
            "pooled_events", "Event agreement (arm entries)",
            ["paradigm", "tolerance_s", "n_videos", "true_positives",
             "false_positives", "false_negatives", "precision", "recall",
             "f1_pooled", "f1_mean_of_videos", "mean_offset_s",
             "mean_abs_offset_s"], event_rows,
            note="Counts are summed over videos, then precision/recall/F1 "
                 "computed from the totals."))

    frame_rows = []
    for paradigm in (PARADIGM_TST, PARADIGM_OPEN_FIELD):
        result = results.paradigms.get(paradigm)
        if result is None:
            continue
        for confusion in result.confusions:
            k = confusion.kappa
            frame_rows.append([
                PARADIGM_LABELS[paradigm], confusion.title, k.n,
                confusion.n_videos, k.kappa, k.ci_low, k.ci_high,
                confusion.percent_agreement
                if math.isfinite(confusion.percent_agreement) else None,
                result.frame_stats.get("immobility_precision"),
                result.frame_stats.get("immobility_recall"),
                result.frame_stats.get("immobility_f1")])
    if frame_rows:
        tables.append(Table(
            "frame_agreement", "Frame-level label agreement",
            ["paradigm", "labels", "frames", "videos", "kappa", "kappa_ci_low",
             "kappa_ci_high", "percent_agreement", "immobility_precision",
             "immobility_recall", "immobility_f1"], frame_rows,
            note="Neighbouring frames are strongly autocorrelated: the frame "
                 "count is not a sample size; judge by video."))
    for paradigm in (PARADIGM_TST, PARADIGM_OPEN_FIELD):
        result = results.paradigms.get(paradigm)
        for confusion in (result.confusions if result else []):
            matrix = confusion.matrix
            tables.append(Table(
                f"{confusion.name}_confusion",
                f"{confusion.title}: confusion matrix (rows automated, "
                "columns manual)",
                ["automated \\ manual"] + list(confusion.categories),
                [[category] + [int(v) for v in matrix[i]]
                 for i, category in enumerate(confusion.categories)]))

    if results.spotcheck_rows:
        tables.append(Table(
            "position_spotcheck_per_video", "Position spot-check: per video",
            results.spotcheck_columns, results.spotcheck_rows))
        p = results.spotcheck
        tables.append(Table(
            "position_spotcheck_pooled", "Position spot-check: pooled",
            ["files", "reviewed", "visible", "not_visible", "matched",
             "missed", "miss_rate", "false_detections",
             "false_detection_rate", "mean_px", "median_px", "p95_px",
             "mean_cm", "median_cm", "p95_cm"],
            [[p.n_files, p.n_reviewed, p.n_visible, p.n_not_visible,
              p.n_matched, p.n_missed, p.miss_rate, p.false_detections,
              p.false_detection_rate, p.mean_px, p.median_px, p.p95_px,
              p.mean_cm, p.median_cm, p.p95_cm]]))

    tables.append(Table(
        "acceptance_criteria", "Acceptance criteria",
        ["criterion", "key", "comparison", "threshold", "value", "result"],
        [[r.label, r.criterion.key, r.criterion.op, r.criterion.threshold,
          r.value, r.status] for r in results.criteria]))
    tables.append(Table(
        "input_files", "Input files",
        ["role", "video", "path", "size_bytes", "sha256"],
        [[f.role, f.stem, f.path, f.size_bytes, f.sha256]
         for f in results.input_files]))
    return tables


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(value) else repr(round(float(value), 8))
    return str(value)


def write_table_csv(path, table: Table) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(table.columns)
        for row in table.rows:
            writer.writerow([_csv_cell(cell) for cell in row])


def build_report(
    inputs: Sequence[VideoInputs],
    config: ReportConfig,
    output_dir,
    name: str = "validation_report",
) -> Dict[str, object]:
    """Compute, then write ``<name>.html`` and one CSV per table.

    Returns ``{"results": ..., "html": Path, "csv": [Path, ...]}``.
    """
    from core.validation_report_render import write_report
    results = build_results(inputs, config)
    html_path, csv_paths = write_report(results, Path(output_dir), name)
    return {"results": results, "html": html_path, "csv": csv_paths}


__all__ = [
    "AGREEMENT_STATS", "ALL_ROLES", "Criterion", "CriterionResult",
    "METRICS", "ReportConfig", "Table", "ValidationResults", "VideoInputs",
    "agreement_statistics", "build_report", "build_results", "build_tables",
    "classify_manual_file", "combine_inputs", "criteria_catalog",
    "default_criteria", "detect_paradigm", "discover_automated",
    "discover_manual", "evaluate_criteria", "flat_statistics",
    "pool_events", "pool_spotchecks", "write_table_csv",
]
