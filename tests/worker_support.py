"""Tiny synthetic videos and settings shared by the worker-level tests."""
import threading
from pathlib import Path

import cv2
import numpy as np

from checkpoint_manager import CheckpointManager
from core.models import BatchSettings, DetectionConfig
from core.pipeline import NullReporter, VideoJob, VideoPipeline
from mazes.tst import TST
from roi_manager import ROIManager
from tests import tst_synthetic as synthetic


def write_tracking_video(
    directory: Path, frames: int = 100, name: str = "blob.avi"
) -> Path:
    """A white square drifting across a dark 160x120 scene at 30 fps."""
    path = Path(directory) / name
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120)
    )
    for index in range(frames):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        cx = 30 + index
        cy = 60 + int(10 * np.sin(index / 9.0))
        cv2.rectangle(
            frame, (cx - 12, cy - 12), (cx + 12, cy + 12),
            (255, 255, 255), -1,
        )
        writer.write(frame)
    writer.release()
    return path


def tracking_settings(create_timelapse: bool = False) -> BatchSettings:
    manager = ROIManager(source_dimensions=(160, 120))
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
        create_timelapse=create_timelapse,
    )


def write_tst_video(directory: Path, duration: float = 4.0) -> Path:
    path = Path(directory) / "tst.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
        synthetic.BASE_SIZE,
    )
    for frame in synthetic.render_clip(30.0, duration=duration):
        writer.write(frame)
    writer.release()
    return path


def tst_settings(create_timelapse: bool = False) -> BatchSettings:
    manager = ROIManager(source_dimensions=synthetic.BASE_SIZE)
    manager.add_roi("mouse_1", synthetic.clip_roi_points())
    return BatchSettings(
        manager, DetectionConfig(), 0.0,
        start_time=0.0, end_time=4.0,
        create_timelapse=create_timelapse,
        per_roi_times={"mouse_1": (0.0, 4.0)},
        paradigm_id="Tail Suspension Test",
        algorithm_parameters=TST().get_algorithm_parameters(),
    )


def make_pipeline(root: Path, maze, settings=None, reporter=None):
    """A ``VideoPipeline`` for calling its phases directly.

    Returns ``(pipeline, cancel)`` where ``cancel`` is the threading.Event
    the pipeline polls; the checkpoint manager is ``pipeline.checkpoint_mgr``.
    """
    root = Path(root)
    output_root = root / "out"
    output_root.mkdir(exist_ok=True)
    job = VideoJob(
        video_path=Path("unused.avi"),
        settings=settings,
        maze_name=maze.name,
        algorithm_parameters=maze.get_algorithm_parameters(),
        output_dir=output_root / "clip",
        base_name="clip",
        checkpoint_root=output_root,
    )
    cancel = threading.Event()
    pipeline = VideoPipeline(
        job, maze, reporter or NullReporter(), cancel,
        CheckpointManager(output_root),
    )
    return pipeline, cancel
