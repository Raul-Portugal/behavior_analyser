"""
core/run_outputs.py
Batch-level run folder helpers shared by the GUI worker and the headless CLI.

Nothing here touches Qt: it creates the timestamped run folder, names every
video's output subfolder, and writes ``run_plan.json`` and ``manifest.json``.
"""
import copy
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from core.batch_plan_keys import build_plan_by_key
from core.models import BatchSettings
from core.version import APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)

# Written into every run folder so the run can be reloaded or resumed later.
RUN_PLAN_FILENAME = "run_plan.json"
MANIFEST_FILENAME = "manifest.json"


def apply_effective_parameters(
    analysis_plan: Mapping[Path, BatchSettings], maze
) -> None:
    """Stamp every settings object with the maze's paradigm and parameters.

    The saved ``run_plan.json`` and the checkpoint hash then describe exactly
    what the run used, whatever the loaded configuration spelled out.
    """
    effective_parameters = maze.get_algorithm_parameters()
    for settings in analysis_plan.values():
        settings.paradigm_id = maze.name
        settings.algorithm_parameters = copy.deepcopy(effective_parameters)


def create_run_folder(output_dir: Path) -> Path:
    """Create ``analysis_run_<timestamp>`` (suffixed if it already exists)."""
    run_suffix = time.strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"analysis_run_{run_suffix}"
    sequence = 2
    while candidate.exists():
        candidate = output_dir / f"analysis_run_{run_suffix}_{sequence}"
        sequence += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_timestamp() -> str:
    """The ISO-style timestamp format used by the manifest."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def output_base_names(video_paths: Sequence[Path]) -> Dict[Path, str]:
    """Output subfolder name per video; duplicate stems get a path hash."""
    stem_counts = Counter(path.stem for path in video_paths)
    names: Dict[Path, str] = {}
    for path in video_paths:
        base_name = path.stem
        if stem_counts[path.stem] > 1:
            path_hash = hashlib.sha256(
                str(path.resolve()).encode('utf-8')
            ).hexdigest()[:8]
            base_name = f"{base_name}_{path_hash}"
        names[path] = base_name
    return names


def write_run_plan(
    analysis_plan: Mapping[Path, BatchSettings], run_dir: Path
) -> None:
    """Save the exact per-video settings of this run as a batch plan.

    Keys are relative to the run folder, so the plan can be reloaded with
    File > Load Configuration or resumed after a restart. The paradigm
    and algorithm parameters are part of every entry. A failure here is
    logged but never stops the analysis.
    """
    plan_file = run_dir / RUN_PLAN_FILENAME
    try:
        BatchSettings.save_batch_plan(
            build_plan_by_key(analysis_plan, plan_file), plan_file
        )
    except Exception:
        logger.warning("Could not write %s", plan_file, exc_info=True)


def write_run_manifest(
    run_dir: Path,
    maze,
    started_at: str,
    requested_videos: int,
    base_names: Mapping[Path, str],
    results: Sequence[Tuple[Path, object]],
    failures: Sequence[Tuple[Path, str]],
    quality_warnings: Sequence[Tuple[Path, str]],
    cancelled: bool,
) -> None:
    """Write a compact index so a run can be understood at a glance."""
    generated_files = [MANIFEST_FILENAME] + sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    manifest = {
        "schema_version": 2,
        "application": {
            "name": APP_NAME,
            "version": APP_VERSION,
        },
        "paradigm": maze.name,
        "started_at": started_at,
        "finished_at": run_timestamp(),
        "cancelled": cancelled,
        "requested_videos": requested_videos,
        "completed_videos": len(results),
        "videos": [
            {
                "source": str(video_path),
                "output_folder": base_names[video_path],
            }
            for video_path, _ in results
        ],
        "failures": [
            {"source": str(path), "message": message}
            for path, message in failures
        ],
        "quality_warnings": [
            {"source": str(path), "message": message}
            for path, message in quality_warnings
        ],
        "algorithm_parameters": maze.get_algorithm_parameters(),
        "generated_files": generated_files,
    }
    manifest_path = run_dir / MANIFEST_FILENAME
    temp_path = manifest_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.flush()
    temp_path.replace(manifest_path)
