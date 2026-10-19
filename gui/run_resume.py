"""
gui/run_resume.py
Find interrupted runs that can be resumed after an application restart.

Every run folder holds a ``run_plan.json`` (a batch plan keyed relative to the
run folder, written by the worker when the run starts). A checkpoint resumes
only when its settings hash matches the settings the next run uses, so the
way back is: read the checkpoint header, find the run plan whose entry for
that video hashes to the same value, and reload that plan. Nothing here needs
the user to remember or reload the exact template.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional

from checkpoint_manager import CheckpointManager
from core.batch_plan_keys import match_videos_to_plan
from core.models import BatchSettings

logger = logging.getLogger(__name__)

RUN_PLAN_FILENAME = "run_plan.json"
RUN_FOLDER_PREFIX = "analysis_run_"
CHECKPOINT_MAX_AGE_DAYS = 7  # mirrors CheckpointManager.load_checkpoint


@dataclass
class InterruptedVideo:
    video: Path
    checkpoint: Path
    last_frame: int
    settings_hash: str


@dataclass
class ResumableRun:
    """A run folder whose plan reproduces the settings of its checkpoints."""
    run_dir: Path
    plan_path: Path
    paradigm_id: Optional[str]
    interrupted: List[InterruptedVideo] = field(default_factory=list)
    # Interrupted videos plus videos of the plan that produced no results.
    pending_videos: List[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.run_dir.name


@dataclass
class ResumeScan:
    runs: List[ResumableRun] = field(default_factory=list)
    # Valid checkpoints that no run plan explains (runs from older versions).
    unmatched: List[InterruptedVideo] = field(default_factory=list)

    @property
    def interrupted_count(self) -> int:
        return sum(len(run.interrupted) for run in self.runs) + len(self.unmatched)

    def __bool__(self) -> bool:
        return bool(self.runs or self.unmatched)


def plan_key_to_path(key: str, plan_file: Path) -> Path:
    """Where a plan key points: absolute as-is, else relative to the plan file."""
    absolute = (
        PurePosixPath(key).is_absolute() or PureWindowsPath(key).is_absolute()
    )
    target = Path(key) if absolute else Path(plan_file).resolve().parent / key
    return Path(os.path.normpath(target))


def read_checkpoint_header(checkpoint_path: Path) -> Optional[dict]:
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            header = json.load(handle)
    except (OSError, ValueError):
        return None
    return header if isinstance(header, dict) else None


def _usable_checkpoints(output_dir: Path) -> List[InterruptedVideo]:
    """Checkpoints that ``load_checkpoint`` could actually resume."""
    found: List[InterruptedVideo] = []
    for checkpoint in sorted(CheckpointManager.find_checkpoints_readonly(output_dir)):
        header = read_checkpoint_header(checkpoint)
        if not header:
            continue
        if header.get("version") != CheckpointManager.CHECKPOINT_VERSION:
            continue
        age_days = (time.time() - float(header.get("timestamp", 0))) / 86400
        if age_days > CHECKPOINT_MAX_AGE_DAYS:
            continue
        try:
            video = Path(header["video_path"])
            settings_hash = str(header["settings_hash"])
            if header.get("video_fingerprint") != CheckpointManager.get_video_fingerprint(video):
                continue
            last_frame = int(header.get("last_frame_processed", 0))
        except (KeyError, OSError, TypeError, ValueError):
            continue
        found.append(InterruptedVideo(video, checkpoint, last_frame, settings_hash))
    return found


def _run_folders(output_dir: Path) -> List[Path]:
    """Run folders that hold a plan, newest first (names sort by timestamp)."""
    try:
        folders = [
            path for path in output_dir.glob(f"{RUN_FOLDER_PREFIX}*")
            if (path / RUN_PLAN_FILENAME).is_file()
        ]
    except OSError:
        return []
    return sorted(folders, key=lambda path: path.name, reverse=True)


def _has_results(run_dir: Path, video: Path) -> bool:
    """True if the run folder holds exported data for ``video``."""
    stem = video.stem
    try:
        for folder in run_dir.iterdir():
            name = folder.name
            is_candidate = name == stem or (
                name.startswith(stem + "_")
                and len(name) == len(stem) + 9
                and all(ch in "0123456789abcdef" for ch in name[len(stem) + 1:])
            )
            if is_candidate and folder.is_dir() and any(
                (folder / "data").glob("*")
            ):
                return True
    except OSError:
        return False
    return False


def scan_for_resumable_runs(output_dir: Path) -> ResumeScan:
    """Pair checkpoints in ``output_dir/.checkpoints`` with their run plans."""
    scan = ResumeScan()
    remaining = _usable_checkpoints(output_dir)
    if not remaining:
        return scan
    hasher = CheckpointManager(output_dir, create_dir=False)

    for run_dir in _run_folders(output_dir):
        if not remaining:
            break
        plan_path = run_dir / RUN_PLAN_FILENAME
        try:
            plan: Dict[str, BatchSettings] = BatchSettings.load_batch_plan(plan_path)
        except Exception:
            logger.warning("Could not read %s", plan_path, exc_info=True)
            continue

        explained: List[InterruptedVideo] = []
        for item in remaining:
            match = match_videos_to_plan([item.video], plan.keys(), plan_path)
            key = match.matched.get(item.video)
            if key is None:
                continue
            if hasher.get_settings_hash(plan[key]) == item.settings_hash:
                explained.append(item)
        if not explained:
            continue

        remaining = [item for item in remaining if item not in explained]
        interrupted_videos = {item.video for item in explained}
        paradigm = next(
            (settings.paradigm_id for settings in plan.values()
             if settings.paradigm_id),
            None,
        )
        pending = list(interrupted_videos)
        for key in plan:
            video = plan_key_to_path(key, plan_path)
            if (
                video not in interrupted_videos
                and video not in pending
                and video.is_file()
                and not _has_results(run_dir, video)
            ):
                pending.append(video)
        scan.runs.append(ResumableRun(
            run_dir=run_dir,
            plan_path=plan_path,
            paradigm_id=paradigm,
            interrupted=explained,
            pending_videos=sorted(pending, key=lambda path: str(path)),
        ))

    scan.unmatched = remaining
    return scan


def describe_scan(scan: ResumeScan) -> str:
    """One-line banner text for the main window."""
    if not scan:
        return ""
    parts = []
    for run in scan.runs:
        count = len(run.interrupted)
        parts.append(
            f"{count} interrupted video{'s' if count != 1 else ''} in {run.name}"
        )
    if scan.unmatched:
        count = len(scan.unmatched)
        parts.append(
            f"{count} from an older run without a saved plan (resumes when "
            "you start with exactly the same settings)"
        )
    return "Interrupted analysis found: " + "; ".join(parts) + "."
