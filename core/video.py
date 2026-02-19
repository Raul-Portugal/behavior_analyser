"""
core/video.py
Unified Video I/O: Metadata, Buffered Reading, and Writing.
Replaces video_io.py.
"""
import cv2
import hashlib
import logging
import pickle
import time
import numpy as np
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Tuple, Generator
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoHandler:
    """
    Central handler for video metadata and basic operations.
    Replaces VideoInfo.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise IOError(f"Video file not found: {self.path}")
            
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
            
        try:
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            # Validation
            if self.width <= 0 or self.height <= 0:
                ret, frame = self._cap.read()
                if ret:
                    self.width, self.height = frame.shape[1], frame.shape[0]
                    # Reset
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        finally:
            self._cap.release()

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.width, self.height

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Random access frame retrieval (slow, for UI)."""
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None


class BufferedVideoReader:
    """
    Thread-safe buffered video reader for sequential processing.
    """
    def __init__(self, path: Path, start_frame: int = 0, buffer_size: int = 64):
        self.path = path
        self.start_frame = start_frame
        self.buffer = Queue(maxsize=buffer_size)
        self.stop_event = Event()
        self.reader_thread = Thread(target=self._worker, daemon=True)
        self.reader_thread.start()

    def _worker(self):
        cap = cv2.VideoCapture(str(self.path))
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self.buffer.put(None) # EOF signal
                break
                
            # Retry put if full to allow check of stop_event
            while not self.stop_event.is_set():
                try:
                    self.buffer.put((ret, frame), timeout=0.1)
                    break
                except: continue
        cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            item = self.buffer.get(timeout=5.0)
            if item is None: return False, None
            return item
        except Empty:
            return False, None

    def release(self):
        self.stop_event.set()
        # Drain queue
        while not self.buffer.empty():
            try: self.buffer.get_nowait()
            except: pass
        self.reader_thread.join(timeout=1.0)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.release()


class ReferenceFrameGenerator:
    """Handles generation and caching of background reference frames."""
    
    CACHE_VERSION = 1
    
    @staticmethod
    def get_hash(path: Path) -> str:
        h = hashlib.md5()
        h.update(str(path.stat().st_size).encode())
        h.update(str(path.stat().st_mtime).encode())
        return h.hexdigest()

    @classmethod
    def generate(cls, path: Path, num_samples: int = 100, 
                 target_dims: Optional[Tuple[int, int]] = None, use_cache: bool = True) -> np.ndarray:
        
        cache_path = path.parent / f".{path.stem}_ref.pkl"
        file_hash = cls.get_hash(path)

        # 1. Try Load Cache
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                if data['hash'] == file_hash and data['dims'] == target_dims:
                    return data['frame']
            except: pass

        # 2. Generate
        handler = VideoHandler(path)
        indices = np.linspace(0, handler.total_frames - 1, min(num_samples, handler.total_frames), dtype=int)
        frames = []
        
        cap = cv2.VideoCapture(str(path))
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if target_dims:
                    gray = cv2.resize(gray, target_dims)
                frames.append(gray)
        cap.release()

        if not frames:
            raise ValueError("Could not read frames for reference")

        ref_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)

        # 3. Save Cache
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'hash': file_hash, 'dims': target_dims, 'frame': ref_frame}, f)
            except: pass
            
        return ref_frame


class SafeVideoWriter:
    """Wrapper for cv2.VideoWriter."""
    def __init__(self, path: Path, fps: float, dims: Tuple[int, int], codec='mp4v'):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, dims)
    
    def write(self, frame):
        if self.writer.isOpened(): self.writer.write(frame)
        
    def release(self):
        if self.writer: self.writer.release()
        
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.release()