"""
Checkpoint management system for resumable analysis.
Allows recovery from crashes or cancellations.

A checkpoint is append-only so the I/O per save is proportional to the
frames added since the previous save, not to the whole run so far:

* ``<stem>_<id>_checkpoint.json`` is a small header written atomically
  (version, settings hash, video fingerprint, last processed frame, temporal
  engine state, scalar result fields and the number of committed rows).
* ``<stem>_<id>_checkpoint.rows.jsonl`` holds one JSON line per processed
  frame.  Rows are appended and fsynced *before* the header is rewritten, so
  the header only ever refers to rows that are fully on disk.  Loading reads
  exactly ``rows_committed`` rows and ignores anything after them, such as a
  partially written trailing line left by a crash.
"""
import logging
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisCheckpoint:
    """Container for analysis checkpoint data."""
    video_path: Path
    last_frame_processed: int
    partial_result: Dict[str, Any]
    settings_hash: str
    video_fingerprint: str
    timestamp: float
    version: int = 4


class _RowStream:
    """In-memory bookkeeping for one video's append-only sidecar."""

    def __init__(self, schema: Dict[str, Any], rows_committed: int,
                 committed_bytes: int):
        self.schema = schema
        self.rows_committed = rows_committed
        self.committed_bytes = committed_bytes


class CheckpointManager:
    """Manages analysis checkpoints for resume functionality."""

    CHECKPOINT_VERSION = 4
    CHECKPOINT_INTERVAL_SECONDS = 30.0  # Wall-clock time between saves
    # Bulky per-frame debugging data that is not needed to resume exactly
    # (the TST validation video simply omits it for restored frames).
    OMITTED_ROW_KEYS = frozenset({"mask_contour"})

    def __init__(self, output_dir: Path,
                 interval_seconds: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic,
                 create_dir: bool = True):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to store checkpoint files
            interval_seconds: Wall-clock seconds between periodic saves
            clock: Monotonic time source (injectable for tests)
            create_dir: Create the ``.checkpoints`` folder. Pass False for
                read-only lookups, which must not touch the filesystem.
        """
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / ".checkpoints"
        if create_dir:
            self.checkpoints_dir.mkdir(exist_ok=True)
        self.interval_seconds = (
            self.CHECKPOINT_INTERVAL_SECONDS
            if interval_seconds is None else float(interval_seconds)
        )
        self._clock = clock
        self._last_save_time = clock()
        self._streams: Dict[Path, _RowStream] = {}
        self.bytes_written = 0

    def _get_checkpoint_path(self, video_path: Path) -> Path:
        """Get checkpoint header path for a video."""
        path_id = hashlib.sha256(
            str(video_path.resolve()).encode('utf-8')
        ).hexdigest()[:12]
        return self.checkpoints_dir / f"{video_path.stem}_{path_id}_checkpoint.json"

    @staticmethod
    def _get_rows_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_suffix('.rows.jsonl')

    def _get_settings_hash(self, settings: Any) -> str:
        """Get hash of settings for validation."""
        if hasattr(settings, 'to_dict'):
            settings_data = settings.to_dict()
        else:
            settings_data = settings.__dict__
        settings_json = json.dumps(
            settings_data,
            sort_keys=True,
            separators=(',', ':'),
            default=self._json_default
        )
        return hashlib.sha256(settings_json.encode('utf-8')).hexdigest()[:24]

    @staticmethod
    def _get_video_fingerprint(video_path: Path) -> str:
        stat = video_path.stat()
        payload = (
            f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]

    @staticmethod
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, 'tolist'):
            return value.tolist()
        if hasattr(value, 'item'):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    # ------------------------------------------------------------------
    # Splitting a result into per-frame rows and scalar header fields
    # ------------------------------------------------------------------
    @staticmethod
    def _split_result(partial_result: Dict[str, Any]):
        """Separate per-frame data from everything else.

        Per-frame data is a list whose length equals the frame count
        (``len(timestamps)``), or a non-empty dict whose values are all such
        lists (the TST per-mouse traces).  Returns ``(frame_count, schema,
        scalars)``.
        """
        frame_count = len(partial_result.get('timestamps') or [])
        list_fields: List[str] = []
        dict_fields: Dict[str, List[str]] = {}
        scalars: Dict[str, Any] = {}
        for name, value in partial_result.items():
            if frame_count and isinstance(value, list) and len(value) == frame_count:
                list_fields.append(name)
            elif (
                frame_count
                and isinstance(value, dict)
                and value
                and all(
                    isinstance(item, list) and len(item) == frame_count
                    for item in value.values()
                )
            ):
                dict_fields[name] = sorted(value)
            else:
                scalars[name] = value
        schema = {'lists': sorted(list_fields), 'dicts': dict_fields}
        return frame_count, schema, scalars

    def _encode_rows(self, partial_result, schema, start: int, stop: int) -> bytes:
        omitted = self.OMITTED_ROW_KEYS
        lines = []
        for index in range(start, stop):
            row: Dict[str, Any] = {}
            for name in schema['lists']:
                row[name] = self._strip(partial_result[name][index], omitted)
            for name, keys in schema['dicts'].items():
                row[name] = {
                    key: self._strip(partial_result[name][key][index], omitted)
                    for key in keys
                }
            lines.append(json.dumps(
                row, separators=(',', ':'), default=self._json_default
            ))
        if not lines:
            return b''
        return ('\n'.join(lines) + '\n').encode('utf-8')

    @staticmethod
    def _strip(value, omitted):
        if isinstance(value, dict) and omitted.intersection(value):
            return {k: v for k, v in value.items() if k not in omitted}
        return value

    @staticmethod
    def _write_synced(path: Path, data: bytes):
        with open(path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def save_checkpoint(self, video_path: Path, frame_idx: int,
                       partial_result: Dict[str, Any], settings: Any):
        """
        Save analysis checkpoint.

        Only the rows added since the previous save are written to the
        sidecar; the header is then replaced atomically.

        Args:
            video_path: Video being analyzed
            frame_idx: Last frame processed
            partial_result: Partial analysis results
            settings: Analysis settings
        """
        checkpoint_path = self._get_checkpoint_path(video_path)
        rows_path = self._get_rows_path(checkpoint_path)

        try:
            frame_count, schema, scalars = self._split_result(partial_result)
            stream = self._streams.get(checkpoint_path)
            # Anything unexpected (no known stream, new field layout, fewer
            # rows than already committed) falls back to a full rewrite.
            append = (
                stream is not None
                and stream.schema == schema
                and stream.rows_committed <= frame_count
                and rows_path.exists()
            )
            if append:
                start, offset = stream.rows_committed, stream.committed_bytes
            else:
                start, offset = 0, 0
            payload = self._encode_rows(partial_result, schema, start, frame_count)

            with open(rows_path, 'r+b' if append else 'wb') as f:
                f.seek(offset)
                # Drop uncommitted leftovers (torn line, failed earlier save).
                f.truncate(offset)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
                committed_bytes = f.tell()

            header = {
                'version': self.CHECKPOINT_VERSION,
                'video_path': str(video_path),
                'last_frame_processed': frame_idx,
                'settings_hash': self._get_settings_hash(settings),
                'video_fingerprint': self._get_video_fingerprint(video_path),
                'timestamp': time.time(),
                'rows_committed': frame_count,
                'row_schema': schema,
                'partial_result': scalars,
            }
            header_bytes = json.dumps(
                header, separators=(',', ':'), default=self._json_default
            ).encode('utf-8')
            temp_path = checkpoint_path.with_suffix('.tmp')
            self._write_synced(temp_path, header_bytes)
            temp_path.replace(checkpoint_path)

            self._streams[checkpoint_path] = _RowStream(
                schema, frame_count, committed_bytes
            )
            self.bytes_written += len(payload) + len(header_bytes)
            self._last_save_time = self._clock()
            logger.debug(f"Checkpoint saved at frame {frame_idx}")

        except (OSError, TypeError, ValueError) as e:
            # The stream state is only advanced after a full save, so the
            # next attempt re-truncates to the last committed offset.
            logger.warning(f"Could not save checkpoint: {e}")

    def load_checkpoint(self, video_path: Path, settings: Any) -> Optional[AnalysisCheckpoint]:
        """
        Load checkpoint if exists and valid.

        Args:
            video_path: Video path
            settings: Current analysis settings

        Returns:
            Checkpoint data or None
        """
        checkpoint_path = self._get_checkpoint_path(video_path)
        self._streams.pop(checkpoint_path, None)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)

            # Validate version
            if checkpoint_dict.get('version') != self.CHECKPOINT_VERSION:
                logger.info("Checkpoint version mismatch, starting fresh")
                return None

            # Validate settings match
            current_hash = self._get_settings_hash(settings)
            if checkpoint_dict.get('settings_hash') != current_hash:
                logger.info("Settings changed, starting fresh")
                return None

            current_video_fingerprint = self._get_video_fingerprint(video_path)
            if checkpoint_dict.get('video_fingerprint') != current_video_fingerprint:
                logger.info("Video changed, starting fresh")
                return None

            # Check if checkpoint is too old (> 7 days)
            age_days = (time.time() - checkpoint_dict.get('timestamp', 0)) / 86400
            if age_days > 7:
                logger.info("Checkpoint too old, starting fresh")
                return None

            schema = checkpoint_dict['row_schema']
            rows_committed = int(checkpoint_dict['rows_committed'])
            partial_result = dict(checkpoint_dict['partial_result'])
            committed_bytes = self._read_rows(
                self._get_rows_path(checkpoint_path),
                schema, rows_committed, partial_result
            )
            if committed_bytes is None:
                logger.info("Checkpoint rows incomplete, starting fresh")
                return None

            checkpoint = AnalysisCheckpoint(
                video_path=Path(checkpoint_dict['video_path']),
                last_frame_processed=int(checkpoint_dict['last_frame_processed']),
                partial_result=partial_result,
                settings_hash=checkpoint_dict['settings_hash'],
                video_fingerprint=checkpoint_dict['video_fingerprint'],
                timestamp=checkpoint_dict['timestamp'],
                version=checkpoint_dict['version'],
            )
            self._streams[checkpoint_path] = _RowStream(
                schema, rows_committed, committed_bytes
            )
            logger.info(f"Found resumable checkpoint at frame {checkpoint.last_frame_processed}")
            return checkpoint

        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return None

    @staticmethod
    def _read_rows(rows_path: Path, schema, rows_committed: int,
                   partial_result: Dict[str, Any]) -> Optional[int]:
        """Rebuild per-frame data from exactly ``rows_committed`` lines.

        Returns the byte offset just past the last committed row, or None if
        the sidecar does not hold that many complete rows.
        """
        lists = {name: [] for name in schema['lists']}
        dicts = {
            name: {key: [] for key in keys}
            for name, keys in schema['dicts'].items()
        }
        offset = 0
        count = 0
        if rows_committed > 0:
            if not rows_path.exists():
                return None
            with open(rows_path, 'rb') as f:
                for line in f:
                    if count == rows_committed:
                        break
                    if not line.endswith(b'\n'):
                        return None
                    row = json.loads(line)
                    for name, values in lists.items():
                        values.append(row[name])
                    for name, traces in dicts.items():
                        for key, values in traces.items():
                            values.append(row[name][key])
                    offset += len(line)
                    count += 1
            if count != rows_committed:
                return None
        partial_result.update(lists)
        partial_result.update(dicts)
        return offset

    def clear_checkpoint(self, video_path: Path):
        """Clear checkpoint after successful completion."""
        checkpoint_path = self._get_checkpoint_path(video_path)
        self._streams.pop(checkpoint_path, None)
        try:
            for path in (
                checkpoint_path,
                self._get_rows_path(checkpoint_path),
                checkpoint_path.with_suffix('.tmp'),
            ):
                if path.exists():
                    path.unlink()
            logger.debug(f"Checkpoint cleared for {video_path.name}")
        except OSError as e:
            logger.warning(f"Could not clear checkpoint: {e}")

    def find_all_checkpoints(self) -> list[Path]:
        """Find all available checkpoints."""
        try:
            return list(self.checkpoints_dir.glob("*_checkpoint.json"))
        except OSError:
            return []

    @classmethod
    def find_checkpoints_readonly(cls, output_dir: Path) -> list[Path]:
        """List checkpoints under ``output_dir`` without creating any folder."""
        return cls(output_dir, create_dir=False).find_all_checkpoints()

    def reset_timer(self):
        """Restart the periodic-save clock (call at the start of a video)."""
        self._last_save_time = self._clock()

    def should_save_checkpoint(self, frame_idx: Optional[int] = None) -> bool:
        """Whether the wall-clock save interval has elapsed.

        ``frame_idx`` is accepted for backward compatibility and ignored.
        """
        return (
            self._clock() - self._last_save_time >= self.interval_seconds
        )


class ProgressEstimator:
    """Accurate progress and time estimation."""
    
    def __init__(self, total_frames: int):
        """
        Initialize estimator.
        
        Args:
            total_frames: Total number of frames to process
        """
        self.total_frames = total_frames
        self.frame_times = []
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def update(self, current_frame: int):
        """
        Update progress estimation.
        
        Args:
            current_frame: Current frame number
        """
        now = time.time()
        frame_time = now - self.last_update
        self.frame_times.append(frame_time)
        
        # Keep only recent measurements (rolling window)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        self.last_update = now
    
    def get_eta_seconds(self, current_frame: int) -> float:
        """
        Get estimated time remaining in seconds.
        
        Args:
            current_frame: Current frame number
            
        Returns:
            Estimated seconds remaining
        """
        if not self.frame_times or current_frame >= self.total_frames:
            return 0.0
        
        # Use recent average for better accuracy
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        frames_left = self.total_frames - current_frame
        
        return frames_left * avg_frame_time
    
    def get_eta_string(self, current_frame: int) -> str:
        """
        Get formatted ETA string.
        
        Args:
            current_frame: Current frame number
            
        Returns:
            Formatted string like "2m 15s" or "Calculating..."
        """
        if len(self.frame_times) < 10:
            return "Calculating..."
        
        seconds = self.get_eta_seconds(current_frame)
        
        if seconds > 3600:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        elif seconds > 60:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            return f"{int(seconds)}s"
    
    def get_speed(self) -> float:
        """
        Get current processing speed in frames/second.
        
        Returns:
            Frames per second
        """
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times[-20:]) / len(self.frame_times[-20:])
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_progress_percentage(self, current_frame: int) -> float:
        """Get progress as percentage."""
        return (current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0.0
