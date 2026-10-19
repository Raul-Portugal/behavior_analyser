"""
gui/workspace/thumbnails.py
First-frame thumbnails and basic video facts (resolution, fps, duration) for
the context panel, read off the GUI thread and cached.

``ThumbnailLoader`` owns one worker thread (a private ``QThreadPool``). Ask
with :meth:`ThumbnailLoader.request`; the ``loaded(path)`` signal fires once
the result is in the cache and :meth:`ThumbnailLoader.get` returns it. A
cached entry is keyed by the file's path, size and modification time, so an
edited video is read again. Nothing here keeps a reference to a widget: the
only connection is the loader's own slot, which Qt drops when the loader is
deleted, so a late result can never reach a dead object.
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
from PyQt6.QtCore import (QCoreApplication, QObject, QRunnable, QThreadPool,
                          pyqtSignal)
from PyQt6.QtGui import QImage

from gui.qt_image import gray_or_bgr_to_qimage

logger = logging.getLogger(__name__)

THUMBNAIL_WIDTH = 320
CACHE_SIZE = 64

CacheKey = Tuple[str, int, int]


@dataclass(frozen=True)
class VideoPreview:
    """What reading one video's first frame found."""
    path: Path
    image: Optional[QImage] = None     # None when the frame could not be read
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0              # seconds; 0 when unknown
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.image is not None


def read_preview(path: Path, max_width: int = THUMBNAIL_WIDTH) -> VideoPreview:
    """Read the first frame and the basic facts of a video (blocking)."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return VideoPreview(path, error="The video cannot be opened")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = capture.read()
    finally:
        capture.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    if not ok or frame is None:
        return VideoPreview(
            path, width=width, height=height, fps=fps, duration=duration,
            error="The first frame cannot be read",
        )
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(
            frame, (max_width, max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return VideoPreview(
        path, image=gray_or_bgr_to_qimage(frame), width=width, height=height,
        fps=fps, duration=duration,
    )


def _cache_key(path: Path) -> Optional[CacheKey]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (os.path.normcase(str(path)), stat.st_size, stat.st_mtime_ns)


class _JobSignals(QObject):
    done = pyqtSignal(object, object)   # (cache key or None, VideoPreview)


class _Job(QRunnable):
    def __init__(self, key: Optional[CacheKey], path: Path):
        super().__init__()
        self.setAutoDelete(False)
        self.key = key
        self.path = path
        self.signals = _JobSignals()

    def run(self) -> None:
        try:
            preview = read_preview(self.path)
        except Exception as exc:   # never let a worker thread die silently
            logger.warning("Thumbnail of %s failed", self.path, exc_info=True)
            preview = VideoPreview(self.path, error=str(exc))
        self.signals.done.emit(self.key, preview)


class ThumbnailLoader(QObject):
    """Cached, off-thread first-frame reader."""

    loaded = pyqtSignal(str)    # str(path) of a video whose preview arrived

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._cache: "OrderedDict[CacheKey, VideoPreview]" = OrderedDict()
        self._jobs: Dict[Optional[CacheKey], _Job] = {}
        self._closed = False

    def get(self, path: Path) -> Optional[VideoPreview]:
        """The cached preview, or None (not read yet / file unreadable)."""
        key = _cache_key(Path(path))
        if key is None:
            return None
        preview = self._cache.get(key)
        if preview is not None:
            self._cache.move_to_end(key)
        return preview

    def request(self, path: Path) -> bool:
        """Start reading ``path`` unless it is cached or already queued.

        Returns True if a read was started. A missing file starts nothing.
        """
        path = Path(path)
        key = _cache_key(path)
        if self._closed or key is None or key in self._cache or key in self._jobs:
            return False
        job = _Job(key, path)
        job.signals.done.connect(self._on_done)
        self._jobs[key] = job
        self._pool.start(job)
        return True

    def _on_done(self, key, preview: VideoPreview) -> None:
        self._jobs.pop(key, None)
        if self._closed:
            return
        self._cache[key] = preview
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        self.loaded.emit(str(preview.path))

    def wait_idle(self, timeout_ms: int = 10000) -> bool:
        """Block until queued reads finish and deliver their results (tests)."""
        finished = self._pool.waitForDone(timeout_ms)
        QCoreApplication.sendPostedEvents(self)
        return finished

    def shutdown(self) -> None:
        """Drop queued reads and wait for the running one; results are ignored."""
        self._closed = True
        self._pool.clear()
        self._pool.waitForDone()
        self._jobs.clear()
