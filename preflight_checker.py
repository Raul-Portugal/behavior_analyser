"""
Pre-flight validation system.
Comprehensive checks before starting analysis to prevent runtime errors.
"""
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2

logger = logging.getLogger(__name__)


class PreflightChecker:
    """
    Comprehensive validation before analysis.
    Catches issues early to prevent runtime failures.
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def validate_all(self, video_paths: List[Path], settings: any, 
                    output_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Run all validation checks.
        
        Args:
            video_paths: List of videos to analyze
            settings: Analysis settings
            output_dir: Output directory
            
        Returns:
            Tuple of (issues, warnings)
            issues = blocking problems
            warnings = non-blocking concerns
        """
        self.issues = []
        self.warnings = []
        
        # Video validation
        self._validate_videos(video_paths)
        
        # Settings validation
        self._validate_settings(settings)
        
        # Output directory validation
        self._validate_output_dir(output_dir)
        
        # Disk space validation
        self._validate_disk_space(video_paths, output_dir)
        
        # System resources validation
        self._validate_system_resources()
        
        return self.issues, self.warnings
    
    def _validate_videos(self, video_paths: List[Path]):
        """Validate all video files."""
        if not video_paths:
            self.issues.append("No videos selected for analysis")
            return
        
        for video_path in video_paths:
            # Check file exists
            if not video_path.exists():
                self.issues.append(f"Video not found: {video_path.name}")
                continue
            
            # Check file size
            size_mb = video_path.stat().st_size / (1024**2)
            if size_mb < 0.1:
                self.warnings.append(f"Video very small ({size_mb:.1f}MB): {video_path.name}")
            elif size_mb > 10000:  # 10GB
                self.warnings.append(f"Video very large ({size_mb:.0f}MB): {video_path.name} - may be slow")
            
            # Try to open video
            can_open, details = self._can_open_video(video_path)
            if not can_open:
                self.issues.append(f"Cannot open video: {video_path.name} - {details}")
            elif details:
                self.warnings.append(f"{video_path.name}: {details}")
    
    def _can_open_video(self, video_path: Path) -> Tuple[bool, str]:
        """
        Test if video can be opened.
        
        Returns:
            (success, details/error)
        """
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return False, "OpenCV cannot open file"
            
            # Try to read first frame
            ret, frame = cap.read()
            if not ret:
                return False, "Cannot read frames"
            
            # Check properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if width <= 0 or height <= 0:
                return False, f"Invalid dimensions: {width}x{height}"
            
            if fps <= 0:
                return False, f"Invalid FPS: {fps}"
            
            # Warnings (not blocking)
            details = []
            if total_frames <= 0:
                details.append("frame count unavailable")
            
            if fps < 10:
                details.append(f"low FPS ({fps:.1f})")
            
            # Check if video is grayscale
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                details.append("grayscale video detected")
            
            # Check brightness
            mean_brightness = frame.mean()
            if mean_brightness < 30:
                details.append("very dark video - detection may be poor")
            elif mean_brightness > 225:
                details.append("very bright video - detection may be poor")
            
            return True, "; ".join(details) if details else ""
            
        except Exception as e:
            return False, str(e)
        finally:
            if cap is not None:
                cap.release()
    
    def _validate_settings(self, settings: any):
        """Validate analysis settings."""
        try:
            # Check detection settings
            if hasattr(settings, 'detection_config'):
                threshold = settings.detection_config.threshold_percentile
                if threshold < 90:
                    self.warnings.append(
                        f"Low detection threshold ({threshold}%) - may detect noise"
                    )
                elif threshold > 99.9:
                    self.warnings.append(
                        f"Very high detection threshold ({threshold}%) - may miss animal"
                    )
            
            # Check scale factor
            if hasattr(settings, 'scale_factor'):
                if settings.scale_factor == 0:
                    self.warnings.append(
                        "No distance calibration - results will be in pixels"
                    )
                elif settings.scale_factor < 0:
                    self.issues.append("Invalid scale factor (negative)")
            
            # Check time range
            if hasattr(settings, 'start_time') and hasattr(settings, 'end_time'):
                if settings.end_time is not None:
                    if settings.end_time <= settings.start_time:
                        self.issues.append(
                            f"Invalid time range: end ({settings.end_time}s) "
                            f"<= start ({settings.start_time}s)"
                        )
            
            # Check ROI manager
            if hasattr(settings, 'roi_manager'):
                if len(settings.roi_manager.rois) == 0:
                    self.warnings.append("No ROIs defined (zone-free mode)")
                
                # Validate each ROI has valid points
                for category, rois in settings.roi_manager.rois.items():
                    for roi in rois:
                        if len(roi.points) < 3:
                            self.issues.append(
                                f"ROI '{category}' has fewer than 3 points"
                            )
            
        except Exception as e:
            logger.warning(f"Error validating settings: {e}")
    
    def _validate_output_dir(self, output_dir: Path):
        """Validate output directory."""
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                self.issues.append(f"Cannot create output directory: {e}")
                return
        
        # Check if writable
        if not os.access(output_dir, os.W_OK):
            self.issues.append(f"Output directory not writable: {output_dir}")
        
        # Check if directory has existing files (warn about overwrite)
        existing_files = list(output_dir.glob("*"))
        if len(existing_files) > 10:
            self.warnings.append(
                f"Output directory contains {len(existing_files)} files - "
                "some may be overwritten"
            )
    
    def _validate_disk_space(self, video_paths: List[Path], output_dir: Path):
        """Validate sufficient disk space."""
        try:
            import psutil
            
            # Estimate required space (rough: 15% of video sizes)
            total_video_size = sum(v.stat().st_size for v in video_paths 
                                  if v.exists())
            required_bytes = total_video_size * 0.15
            required_gb = required_bytes / (1024**3)
            
            # Check available space
            usage = psutil.disk_usage(str(output_dir))
            available_gb = usage.free / (1024**3)
            
            if available_gb < required_gb:
                self.issues.append(
                    f"Insufficient disk space: {available_gb:.1f}GB available, "
                    f"~{required_gb:.1f}GB required"
                )
            elif available_gb < required_gb * 2:
                self.warnings.append(
                    f"Low disk space: {available_gb:.1f}GB available, "
                    f"~{required_gb:.1f}GB required"
                )
            
        except ImportError:
            logger.debug("psutil not available, skipping disk space check")
        except Exception as e:
            logger.warning(f"Error checking disk space: {e}")
    
    def _validate_system_resources(self):
        """Validate system has adequate resources."""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 1:
                self.issues.append(
                    f"Very low memory: {available_gb:.1f}GB available - "
                    "analysis may fail"
                )
            elif available_gb < 2:
                self.warnings.append(
                    f"Low memory: {available_gb:.1f}GB available - "
                    "may experience slowdowns"
                )
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                self.warnings.append(
                    f"High CPU usage ({cpu_percent}%) - analysis may be slow"
                )
            
        except ImportError:
            logger.debug("psutil not available, skipping system resource check")
        except Exception as e:
            logger.warning(f"Error checking system resources: {e}")
    
    def can_proceed(self) -> bool:
        """Check if analysis can proceed (no blocking issues)."""
        return len(self.issues) == 0
    
    def get_summary(self) -> str:
        """Get formatted summary of validation results."""
        lines = []
        
        if self.issues:
            lines.append("❌ BLOCKING ISSUES:")
            for issue in self.issues:
                lines.append(f"  • {issue}")
        
        if self.warnings:
            if lines:
                lines.append("")
            lines.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
        
        if not self.issues and not self.warnings:
            lines.append("✓ All checks passed - ready for analysis")
        
        return "\n".join(lines)


class ValidationDialog:
    """Helper to format validation results for GUI."""
    
    @staticmethod
    def format_for_messagebox(issues: List[str], warnings: List[str]) -> Tuple[str, str]:
        """
        Format validation results for QMessageBox.
        
        Returns:
            (title, message)
        """
        if issues:
            title = "Cannot Start Analysis"
            message = "The following issues must be fixed:\n\n"
            message += "\n".join(f"❌ {issue}" for issue in issues)
            
            if warnings:
                message += "\n\nAdditional warnings:\n"
                message += "\n".join(f"⚠️  {warning}" for warning in warnings[:3])
                if len(warnings) > 3:
                    message += f"\n... and {len(warnings) - 3} more"
            
            return title, message
        
        elif warnings:
            title = "Ready with Warnings"
            message = "Analysis can proceed, but note:\n\n"
            message += "\n".join(f"⚠️  {warning}" for warning in warnings)
            message += "\n\nContinue anyway?"
            
            return title, message
        
        else:
            title = "Pre-flight Check Passed"
            message = "✓ All validation checks passed\n\nReady to start analysis"
            return title, message