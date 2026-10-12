"""
core/selftest.py
Packaged-build self-test for parallel batches (``--selftest-parallel``).

Writes two tiny synthetic videos, analyses them in two 'spawn' child
processes through the real ``BatchRunner`` and checks the outputs. A frozen
executable running this proves that process-pool children start correctly in
the packaged app (``multiprocessing.freeze_support`` and all).
"""
import contextlib
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.batch_runner import BatchRunner
from core.models import BatchSettings, DetectionConfig
from core.run_outputs import (
    apply_effective_parameters, create_run_folder, output_base_names,
    run_timestamp,
)
from mazes.freestyle import Freestyle
from roi_manager import ROIManager

VIDEO_SIZE = (160, 120)
FRAMES = (45, 60)


class _Collector:
    """Batch reporter that keeps the log and per-video statuses."""

    def __init__(self):
        self.logs: List[str] = []
        self.statuses: List[Tuple[str, str]] = []

    def progress(self, pct, main, sub=""):
        pass

    def log(self, message):
        self.logs.append(message)

    def video_status(self, video_path, status, message):
        self.statuses.append((Path(video_path).name, status))


def _write_video(path: Path, frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, VIDEO_SIZE
    )
    for index in range(frames):
        frame = np.full((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), 40, dtype=np.uint8)
        cx = 30 + index
        cv2.rectangle(
            frame, (cx - 12, 48), (cx + 12, 72), (255, 255, 255), -1
        )
        writer.write(frame)
    writer.release()


def _settings() -> BatchSettings:
    manager = ROIManager(source_dimensions=VIDEO_SIZE)
    manager.add_roi(
        "arena", np.array([[0, 0], [159, 0], [159, 119], [0, 119]])
    )
    return BatchSettings(
        manager,
        DetectionConfig(
            threshold_percentile=95.0, use_weighting=False,
            min_component_area=20, morphology_kernel_size=3,
            min_detection_confidence=0.1,
        ),
        0.0,
    )


def run_parallel_selftest(report_path: Optional[Path] = None) -> int:
    """Analyse two videos on two worker processes; 0 means everything worked."""
    lines: List[str] = []
    ok = False
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            # Children inherit the environment: keep the reference-frame
            # cache inside the temporary folder.
            saved = {
                key: os.environ.get(key)
                for key in ("LOCALAPPDATA", "XDG_CACHE_HOME")
            }
            os.environ["LOCALAPPDATA"] = str(root / "cache")
            os.environ["XDG_CACHE_HOME"] = str(root / "cache")
            try:
                videos = []
                for index, frames in enumerate(FRAMES):
                    video = root / f"selftest_{index}.avi"
                    _write_video(video, frames)
                    videos.append(video)
                maze = Freestyle()
                plan = {video: _settings() for video in videos}
                apply_effective_parameters(plan, maze)
                run_dir = create_run_folder(root / "out")
                reporter = _Collector()
                outcome = BatchRunner(
                    plan, root / "out", run_dir, maze,
                    output_base_names(videos), reporter,
                    _NeverCancelled(), workers=2,
                    started_at=run_timestamp(),
                ).run()
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            summary = run_dir / "batch_summary.csv"
            lines.append(f"results={len(outcome.results)} "
                         f"failures={outcome.failures} "
                         f"cancelled={outcome.cancelled}")
            lines.append(f"statuses={reporter.statuses}")
            ran_in_parallel = any(
                "2 parallel workers" in line for line in reporter.logs
            )
            ok = (
                len(outcome.results) == len(videos)
                and not outcome.failures
                and not outcome.cancelled
                and summary.exists()
                and ran_in_parallel
            )
    except Exception as exc:  # report instead of crashing a windowed exe
        lines.append(f"selftest raised {type(exc).__name__}: {exc}")
    lines.insert(0, "selftest-parallel: " + ("OK" if ok else "FAILED"))
    text = "\n".join(lines)
    with contextlib.suppress(Exception):
        print(text)
    if report_path is not None:
        with contextlib.suppress(OSError):
            Path(report_path).write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


class _NeverCancelled:
    def is_set(self) -> bool:
        return False
