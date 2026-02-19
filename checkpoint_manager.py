"""
Checkpoint management system for resumable analysis.
Allows recovery from crashes or cancellations.
"""
import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AnalysisCheckpoint:
    """Container for analysis checkpoint data."""
    video_path: Path
    last_frame_processed: int
    partial_result: Dict[str, Any]
    settings_hash: str
    timestamp: float
    version: int = 1


class CheckpointManager:
    """Manages analysis checkpoints for resume functionality."""
    
    CHECKPOINT_VERSION = 1
    CHECKPOINT_INTERVAL = 300  # Save every 300 frames
    
    def __init__(self, output_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to store checkpoint files
        """
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / ".checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
    
    def _get_checkpoint_path(self, video_path: Path) -> Path:
        """Get checkpoint file path for a video."""
        return self.checkpoints_dir / f"{video_path.stem}_checkpoint.pkl"
    
    def _get_settings_hash(self, settings: Any) -> str:
        """Get hash of settings for validation."""
        import hashlib
        settings_str = str(settings.__dict__)
        return hashlib.md5(settings_str.encode()).hexdigest()[:16]
    
    def save_checkpoint(self, video_path: Path, frame_idx: int,
                       partial_result: Dict[str, Any], settings: Any):
        """
        Save analysis checkpoint.
        
        Args:
            video_path: Video being analyzed
            frame_idx: Last frame processed
            partial_result: Partial analysis results
            settings: Analysis settings
        """
        checkpoint_path = self._get_checkpoint_path(video_path)
        
        try:
            checkpoint = AnalysisCheckpoint(
                video_path=video_path,
                last_frame_processed=frame_idx,
                partial_result=partial_result,
                settings_hash=self._get_settings_hash(settings),
                timestamp=time.time(),
                version=self.CHECKPOINT_VERSION
            )
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(asdict(checkpoint), f)
            
            logger.debug(f"Checkpoint saved at frame {frame_idx}")
            
        except Exception as e:
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
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_dict = pickle.load(f)
            
            # Validate version
            if checkpoint_dict.get('version') != self.CHECKPOINT_VERSION:
                logger.info("Checkpoint version mismatch, starting fresh")
                return None
            
            # Validate settings match
            current_hash = self._get_settings_hash(settings)
            if checkpoint_dict.get('settings_hash') != current_hash:
                logger.info("Settings changed, starting fresh")
                return None
            
            # Check if checkpoint is too old (> 7 days)
            age_days = (time.time() - checkpoint_dict.get('timestamp', 0)) / 86400
            if age_days > 7:
                logger.info("Checkpoint too old, starting fresh")
                return None
            
            checkpoint = AnalysisCheckpoint(**checkpoint_dict)
            logger.info(f"✓ Found resumable checkpoint at frame {checkpoint.last_frame_processed}")
            return checkpoint
            
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self, video_path: Path):
        """Clear checkpoint after successful completion."""
        checkpoint_path = self._get_checkpoint_path(video_path)
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Checkpoint cleared for {video_path.name}")
        except Exception as e:
            logger.warning(f"Could not clear checkpoint: {e}")
    
    def find_all_checkpoints(self) -> list[Path]:
        """Find all available checkpoints."""
        try:
            return list(self.checkpoints_dir.glob("*_checkpoint.pkl"))
        except:
            return []
    
    def should_save_checkpoint(self, frame_idx: int) -> bool:
        """Check if checkpoint should be saved at this frame."""
        return frame_idx % self.CHECKPOINT_INTERVAL == 0


class ResourceManager:
    """Smart resource management for optimal performance."""
    
    @staticmethod
    def get_optimal_workers() -> int:
        """
        Calculate optimal number of parallel workers.
        
        Returns:
            Recommended number of workers
        """
        try:
            import psutil
            
            cpu_count = psutil.cpu_count(logical=False) or 4
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Don't max out CPU
            optimal_workers = max(1, cpu_count - 1)
            
            # Reduce if low memory (need ~2GB per worker)
            if available_memory_gb < optimal_workers * 2:
                optimal_workers = max(1, int(available_memory_gb / 2))
            
            # Cap at 8 workers to avoid diminishing returns
            optimal_workers = min(optimal_workers, 8)
            
            logger.info(f"Optimal workers: {optimal_workers} (CPU: {cpu_count}, "
                       f"RAM: {available_memory_gb:.1f}GB)")
            
            return optimal_workers
            
        except ImportError:
            logger.warning("psutil not available, using conservative defaults")
            return 2
        except Exception as e:
            logger.warning(f"Error calculating workers: {e}")
            return 2
    
    @staticmethod
    def check_disk_space(output_dir: Path, required_gb: float = 1.0) -> bool:
        """
        Check if sufficient disk space available.
        
        Args:
            output_dir: Output directory
            required_gb: Required space in GB
            
        Returns:
            True if sufficient space
        """
        try:
            import psutil
            
            usage = psutil.disk_usage(str(output_dir))
            available_gb = usage.free / (1024**3)
            
            if available_gb < required_gb:
                logger.warning(f"Low disk space: {available_gb:.1f}GB available, "
                             f"{required_gb:.1f}GB required")
                return False
            
            return True
            
        except ImportError:
            logger.warning("Cannot check disk space (psutil not available)")
            return True
        except Exception as e:
            logger.warning(f"Error checking disk space: {e}")
            return True
    
    @staticmethod
    def estimate_output_size(video_path: Path) -> float:
        """
        Estimate output size in MB.
        
        Args:
            video_path: Input video path
            
        Returns:
            Estimated size in MB
        """
        try:
            video_size_mb = video_path.stat().st_size / (1024**2)
            # Rough estimate: outputs are ~10-20% of video size
            estimated_mb = video_size_mb * 0.15
            return estimated_mb
        except:
            return 50.0  # Default estimate
    
    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage percentage.
        
        Returns:
            Memory usage as percentage (0-100)
        """
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    @staticmethod
    def should_reduce_memory_usage() -> bool:
        """Check if memory usage is too high."""
        usage = ResourceManager.get_memory_usage()
        return usage > 85.0


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