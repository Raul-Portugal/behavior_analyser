"""Pure helpers for keying and matching videos in saved batch plans.

Plans used to be keyed by bare filename, so two ``mouse1.mp4`` in different
folders collided. Keys are now the video path relative to the plan file's
directory (POSIX separators), or the absolute path when no relative path
exists (another drive). Old filename-keyed plans still match when the
filename is unambiguous on both sides.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Iterable, List, Mapping, Tuple, TypeVar

T = TypeVar("T")


def _norm(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _is_absolute_key(key: str) -> bool:
    return PurePosixPath(key).is_absolute() or PureWindowsPath(key).is_absolute()


def _key_name(key: str) -> str:
    return PureWindowsPath(key.replace("/", "\\")).name


def plan_key_for(video_path: Path, plan_file: Path) -> str:
    """Return the plan key for a video: relative to the plan file's folder."""
    video = Path(video_path).resolve()
    plan_dir = Path(plan_file).resolve().parent
    try:
        return Path(os.path.relpath(video, plan_dir)).as_posix()
    except ValueError:  # different drive
        return video.as_posix()


def build_plan_by_key(
    analysis_plan: Mapping[Path, T], plan_file: Path
) -> Dict[str, T]:
    """Key a {video path: settings} plan for ``BatchSettings.save_batch_plan``."""
    return {
        plan_key_for(video, plan_file): settings
        for video, settings in analysis_plan.items()
    }


@dataclass
class PlanMatch:
    """Outcome of matching selected videos against a loaded plan."""
    matched: Dict[Path, str] = field(default_factory=dict)
    unmatched: List[Tuple[Path, str]] = field(default_factory=list)


def match_videos_to_plan(
    video_paths: Iterable[Path],
    plan_keys: Iterable[str],
    plan_file: Path,
) -> PlanMatch:
    """Match videos to plan keys without ever guessing.

    Order: exact resolved path, then path relative to the plan file, then a
    filename that is unique among both the unused plan keys and the selected
    videos. Anything ambiguous is reported as unmatched with a reason.
    """
    videos = [Path(v) for v in video_paths]
    keys = list(plan_keys)
    plan_dir = Path(plan_file).resolve().parent
    result = PlanMatch()
    used_keys = set()

    # Steps 1 and 2: absolute keys, then keys relative to the plan file.
    resolved_keys: Dict[str, str] = {}
    for key in keys:
        target = Path(key) if _is_absolute_key(key) else plan_dir / key
        resolved_keys[key] = _norm(target)
    for video in videos:
        video_norm = _norm(video)
        for key in keys:
            if key not in used_keys and resolved_keys[key] == video_norm:
                result.matched[video] = key
                used_keys.add(key)
                break

    # Step 3: unique filename on both sides.
    keys_by_name: Dict[str, List[str]] = {}
    for key in keys:
        if key not in used_keys:
            keys_by_name.setdefault(_key_name(key).lower(), []).append(key)
    videos_by_name: Dict[str, int] = {}
    for video in videos:
        videos_by_name[video.name.lower()] = (
            videos_by_name.get(video.name.lower(), 0) + 1
        )

    for video in videos:
        if video in result.matched:
            continue
        name = video.name.lower()
        candidates = keys_by_name.get(name, [])
        if not candidates:
            result.unmatched.append((video, "not in the plan"))
        elif len(candidates) > 1:
            result.unmatched.append((
                video,
                f"filename matches {len(candidates)} plan entries",
            ))
        elif videos_by_name[name] > 1:
            result.unmatched.append((
                video,
                "several selected videos share this filename",
            ))
        else:
            result.matched[video] = candidates[0]
    return result
