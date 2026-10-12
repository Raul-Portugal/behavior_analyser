"""Engine runs over the synthetic TST clips, computed once per test process.

Rendering a clip and pushing it through the engine dominates the TST test
time, and several modules need the same observations. This module renders the
15/30/60 fps clips and runs each engine variant on a thread pool (OpenCV
releases the GIL), then caches the result for every test that asks.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from core.tst_motion_engine import TSTMotionEngine
from roi_manager import ROI
from tests import tst_synthetic as synthetic
from tests.tst_reference_engine import ReferenceTSTMotionEngine

RATES = (15.0, 30.0, 60.0)
MOUSE = "mouse_1"

_lock = threading.Lock()
_frames = {}
_runs = {}


def clip_roi() -> ROI:
    return ROI(synthetic.clip_roi_points(), MOUSE)


def serial_run(engine_class, frames, fps, rois=None):
    """Observations of every frame pair, one ``calculate_observations`` each."""
    engine = engine_class(rois or {MOUSE: clip_roi()}, fps=fps)
    return [
        engine.calculate_observations(frames[index], frames[index - 1])
        for index in range(1, len(frames))
    ]


def pipelined_run(frames, fps, rois=None, **options):
    """The same observations through :class:`TSTFramePipeline`."""
    from core.tst_motion_engine import TSTFramePipeline

    engine = TSTMotionEngine(rois or {MOUSE: clip_roi()}, fps=fps)
    results = []
    with TSTFramePipeline(engine, **options) as pipeline:
        for index in range(1, len(frames)):
            while not pipeline.has_capacity():
                results.append(pipeline.pop()[1])
            pipeline.submit(index, frames[index], frames[index - 1])
        while pipeline.pending:
            results.append(pipeline.pop()[1])
    return results


_RUNNERS = {
    "reference": lambda frames, fps: serial_run(
        ReferenceTSTMotionEngine, frames, fps
    ),
    "serial": lambda frames, fps: serial_run(TSTMotionEngine, frames, fps),
    "pipelined": pipelined_run,
}


def clip_frames(fps):
    with _lock:
        frames = _frames.get(fps)
    if frames is None:
        frames = synthetic.render_clip(fps)
        with _lock:
            frames = _frames.setdefault(fps, frames)
    return frames


def runs(*kinds):
    """Return ``{kind: {fps: [per-pair {mouse: observation}]}}``."""
    with _lock:
        missing = [
            (kind, fps) for kind in kinds for fps in RATES
            if (kind, fps) not in _runs
        ]
    if missing:
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
            list(pool.map(clip_frames, RATES))
            futures = {
                job: pool.submit(_RUNNERS[job[0]], clip_frames(job[1]), job[1])
                for job in missing
            }
            computed = {job: future.result() for job, future in futures.items()}
        with _lock:
            _runs.update(computed)
    with _lock:
        return {
            kind: {fps: _runs[(kind, fps)] for fps in RATES} for kind in kinds
        }
