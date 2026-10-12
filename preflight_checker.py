"""
Pre-flight validation system.
Comprehensive checks before starting analysis to prevent runtime errors.

One batch-level entry point (``PreflightChecker.validate_batch``) runs the
per-video checks once per video (each video is opened a single time) and the
global checks (memory, output directory, disk) once for the whole batch.
Every check reports its own failure as a blocking issue instead of silently
skipping the checks that follow it.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

import cv2

from mazes.tst import TST

logger = logging.getLogger(__name__)

# Rough output footprint (plots, tables, optional media) relative to inputs.
DISK_BYTES_FRACTION = 0.15
# A last-frame timestamp this far from frames/fps suggests variable frame rate.
VFR_RELATIVE_TOLERANCE = 0.01
VFR_MIN_FRAME_TOLERANCE = 5


# ============================================================================
# PURE HELPERS
# ============================================================================

def estimate_required_bytes(video_sizes: Sequence[int]) -> float:
    """Estimated output footprint for a whole batch (summed across videos)."""
    return float(sum(video_sizes)) * DISK_BYTES_FRACTION


def disk_space_findings(
    required_bytes: float, free_bytes: float
) -> Tuple[Optional[str], Optional[str]]:
    """Return (issue, warning) for a batch-wide disk requirement."""
    required_gb = required_bytes / (1024 ** 3)
    available_gb = free_bytes / (1024 ** 3)
    if free_bytes < required_bytes:
        return (
            f"Insufficient disk space for the whole batch: "
            f"{available_gb:.1f}GB available, ~{required_gb:.1f}GB required",
            None,
        )
    if free_bytes < required_bytes * 2:
        return (
            None,
            f"Low disk space for the whole batch: {available_gb:.1f}GB "
            f"available, ~{required_gb:.1f}GB required",
        )
    return None, None


def vfr_warning(
    declared_duration_s: Optional[float],
    measured_duration_s: Optional[float],
    fps: float,
) -> Optional[str]:
    """Warn when frame-index timing disagrees with the container timestamps.

    ``declared`` is frame_count / fps; ``measured`` comes from the timestamp
    of the last frame. Analysis converts frame indexes to seconds with the
    nominal fps, so a large disagreement means times may drift.
    """
    if not declared_duration_s or not measured_duration_s or fps <= 0:
        return None
    difference = abs(measured_duration_s - declared_duration_s)
    tolerance = max(
        VFR_MIN_FRAME_TOLERANCE / fps,
        VFR_RELATIVE_TOLERANCE * declared_duration_s,
    )
    if difference <= tolerance:
        return None
    return (
        f"likely variable frame rate: {declared_duration_s:.1f}s from frame "
        f"count/FPS but {measured_duration_s:.1f}s from timestamps; analysis "
        "times may drift (consider re-encoding at a constant frame rate)"
    )


@dataclass
class VideoProbe:
    """Everything the per-video checks need, gathered in one open."""
    path: Path
    size_bytes: int = 0
    exists: bool = True
    opened: bool = False
    error: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    mean_brightness: Optional[float] = None
    measured_duration_s: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return None


def probe_video(video_path: Path) -> VideoProbe:
    """Open a video once and collect metadata, brightness and last timestamp."""
    probe = VideoProbe(path=Path(video_path))
    if not probe.path.exists():
        probe.exists = False
        return probe
    probe.size_bytes = probe.path.stat().st_size

    cap = None
    try:
        cap = cv2.VideoCapture(str(probe.path))
        if not cap.isOpened():
            probe.error = "OpenCV cannot open file"
            return probe
        ret, frame = cap.read()
        if not ret:
            probe.error = "Cannot read frames"
            return probe

        probe.opened = True
        probe.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        probe.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.fps = float(cap.get(cv2.CAP_PROP_FPS))
        probe.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.mean_brightness = float(frame.mean())

        # One seek to the last frame; its timestamp exposes VFR files.
        if probe.fps > 0 and probe.frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, probe.frame_count - 1)
            ret, _ = cap.read()
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ret and msec > 0:
                probe.measured_duration_s = msec / 1000.0 + 1.0 / probe.fps
    except Exception as exc:
        probe.opened = False
        probe.error = str(exc)
    finally:
        if cap is not None:
            cap.release()
    return probe


class _Findings:
    """Collects messages, merging identical ones reported for many videos."""

    def __init__(self):
        self._issues: Dict[str, List[str]] = {}
        self._warnings: Dict[str, List[str]] = {}

    def issue(self, message: str, video: Optional[str] = None) -> None:
        self._add(self._issues, message, video)

    def warning(self, message: str, video: Optional[str] = None) -> None:
        self._add(self._warnings, message, video)

    @staticmethod
    def _add(store: Dict[str, List[str]], message: str, video: Optional[str]):
        names = store.setdefault(message, [])
        if video is not None and video not in names:
            names.append(video)

    @staticmethod
    def _render(store: Dict[str, List[str]]) -> List[str]:
        rendered = []
        for message, names in store.items():
            if not names:
                rendered.append(message)
            elif len(names) == 1:
                rendered.append(f"{names[0]}: {message}")
            else:
                shown = ", ".join(names[:3])
                extra = f", +{len(names) - 3} more" if len(names) > 3 else ""
                rendered.append(
                    f"{message} (affects {len(names)} videos: {shown}{extra})"
                )
        return rendered

    def render(self) -> Tuple[List[str], List[str]]:
        return self._render(self._issues), self._render(self._warnings)


# ============================================================================
# CHECKER
# ============================================================================

class PreflightChecker:
    """
    Comprehensive validation before analysis.
    Catches issues early to prevent runtime failures.
    """

    def __init__(self):
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self, video_paths: Sequence[Path], settings: any,
                     output_dir: Path,
                     paradigm_name: Optional[str] = None
                     ) -> Tuple[List[str], List[str]]:
        """Validate one settings object applied to every listed video."""
        return self.validate_batch(
            {Path(v): settings for v in video_paths},
            output_dir,
            paradigm_name=paradigm_name,
        )

    def validate_batch(
        self,
        analysis_plan: Mapping[Path, any],
        output_dir: Optional[Path],
        paradigm_name: Optional[str] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
        prober: Callable[[Path], VideoProbe] = probe_video,
    ) -> Tuple[List[str], List[str]]:
        """
        Run all checks once for a whole batch.

        Args:
            analysis_plan: {video path: that video's settings}
            output_dir: Output directory
            paradigm_name: Selected maze name; used when a settings object
                has no paradigm id (legacy configs)
            progress: Optional callback (done, total, video name)
            prober: Video metadata reader (replaceable in tests)

        Returns:
            Tuple of (issues, warnings); issues are blocking.
        """
        findings = _Findings()
        videos = list(analysis_plan.items())
        if not videos:
            findings.issue("No videos selected for analysis")

        sizes: List[int] = []
        for index, (video_path, settings) in enumerate(videos):
            if progress:
                progress(index, len(videos), Path(video_path).name)
            name = Path(video_path).name
            is_tst = self._is_tst(settings, paradigm_name)

            probe_holder: List[VideoProbe] = []
            self._run_check(
                findings, f"open video {name}", name,
                lambda f, v=video_path: probe_holder.append(prober(Path(v))),
            )
            probe = probe_holder[0] if probe_holder else None
            if probe is not None:
                sizes.append(probe.size_bytes)
                self._run_check(
                    findings, "video properties", name,
                    lambda f, p=probe: self._check_probe(f, p, name),
                )
                self._run_check(
                    findings, "ROI geometry and time range", name,
                    lambda f, p=probe, s=settings: self._check_geometry_and_times(
                        f, p, s, name
                    ),
                )
            self._run_settings_checks(findings, settings, is_tst, name)

        if progress:
            progress(len(videos), len(videos), "")

        self._run_check(
            findings, "calibration consistency", None,
            lambda f: self._check_mixed_calibration(
                f, [(s, self._is_tst(s, paradigm_name)) for _, s in videos]
            ),
        )
        self._run_check(
            findings, "output directory", None,
            lambda f: self._check_output_dir(f, output_dir),
        )
        self._run_check(
            findings, "disk space", None,
            lambda f: self._check_disk_space(f, sizes, output_dir),
        )
        self._run_check(
            findings, "system resources", None, self._check_system_resources
        )

        self.issues, self.warnings = findings.render()
        return self.issues, self.warnings

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tst(settings: any, paradigm_name: Optional[str]) -> bool:
        paradigm = getattr(settings, 'paradigm_id', None) or paradigm_name
        return paradigm == TST.name

    @staticmethod
    def _run_check(findings: _Findings, label: str, video: Optional[str],
                   check: Callable[[_Findings], None]) -> None:
        """Run one check; a crash becomes a blocking issue, not a skip."""
        try:
            check(findings)
        except Exception as exc:
            logger.warning("Pre-flight check '%s' failed", label, exc_info=True)
            findings.issue(f"Pre-flight check '{label}' failed: {exc}", video)

    def _run_settings_checks(self, findings: _Findings, settings: any,
                             is_tst: bool, video: str) -> None:
        for label, check in (
            ("detection settings", self._check_detection),
            ("calibration", self._check_calibration),
            ("time range", self._check_time_range),
            ("ROI definitions", self._check_roi_definitions),
        ):
            self._run_check(
                findings, label, video,
                lambda f, c=check: c(f, settings, is_tst, video),
            )

    # ------------------------------------------------------------------
    # Per-video checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_probe(findings: _Findings, probe: VideoProbe, name: str):
        if not probe.exists:
            findings.issue("Video not found", name)
            return

        size_mb = probe.size_bytes / (1024 ** 2)
        if size_mb < 0.1:
            findings.warning(f"Video very small ({size_mb:.1f}MB)", name)
        elif size_mb > 10000:  # 10GB
            findings.warning(
                f"Video very large ({size_mb:.0f}MB) - may be slow", name
            )

        if not probe.opened:
            findings.issue(f"Cannot open video - {probe.error}", name)
            return

        if probe.width <= 0 or probe.height <= 0:
            findings.issue(
                f"Invalid dimensions: {probe.width}x{probe.height}", name
            )
        if probe.fps <= 0:
            findings.issue(f"Invalid FPS: {probe.fps}", name)
            return

        details = []
        if probe.frame_count <= 0:
            details.append("frame count unavailable")
        if probe.fps < 10:
            details.append(f"low FPS ({probe.fps:.1f})")
        if probe.mean_brightness is not None:
            if probe.mean_brightness < 30:
                details.append("very dark video - detection may be poor")
            elif probe.mean_brightness > 225:
                details.append("very bright video - detection may be poor")
        vfr = vfr_warning(
            probe.duration_s, probe.measured_duration_s, probe.fps
        )
        if vfr:
            details.append(vfr)
        for detail in details:
            findings.warning(detail, name)

    @staticmethod
    def _check_geometry_and_times(findings: _Findings, probe: VideoProbe,
                                  settings: any, name: str):
        """Ensure pixel ROIs and time ranges fit this particular video."""
        if not probe.opened or not hasattr(settings, 'roi_manager'):
            return
        roi_manager = settings.roi_manager
        has_pixel_geometry = (
            bool(roi_manager.rois)
            or roi_manager.reference_length_pixels > 0
            or getattr(roi_manager, 'arena', None) is not None
        )
        source_dims = roi_manager.source_dimensions
        if (
            has_pixel_geometry
            and source_dims is not None
            and (probe.width, probe.height) != tuple(source_dims)
        ):
            findings.issue(
                f"ROI geometry is {source_dims[0]}x{source_dims[1]}, but the "
                f"video is {probe.width}x{probe.height}. "
                "Align the template to this video (Tools > Align template to "
                "videos... or 'Check alignment' while verifying), redraw the "
                "ROIs, or use a video with matching dimensions.",
                name,
            )

        duration = probe.duration_s
        if duration is None:
            return
        slack = 1.0 / probe.fps
        start_time = getattr(settings, 'start_time', 0.0)
        end_time = getattr(settings, 'end_time', None)
        if start_time >= duration:
            findings.issue(
                f"analysis start ({start_time:.2f}s) is outside the "
                f"{duration:.2f}s video", name,
            )
        if end_time is not None and end_time > duration + slack:
            findings.issue(
                f"analysis end ({end_time:.2f}s) exceeds the "
                f"{duration:.2f}s video", name,
            )
        for roi_name, interval in getattr(settings, 'per_roi_times', {}).items():
            if interval[1] > duration + slack:
                findings.issue(
                    f"{roi_name} end ({interval[1]:.2f}s) exceeds the "
                    f"{duration:.2f}s video", name,
                )

    # ------------------------------------------------------------------
    # Settings checks (deduplicated across videos by _Findings)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_detection(findings: _Findings, settings: any, is_tst: bool,
                         video: str):
        # TST uses its own motion engine, not the centroid detector.
        if is_tst or not hasattr(settings, 'detection_config'):
            return
        threshold = settings.detection_config.threshold_percentile
        if threshold < 90:
            findings.warning(
                f"Low detection threshold ({threshold}%) - may detect noise",
                video,
            )
        elif threshold > 99.9:
            findings.warning(
                f"Very high detection threshold ({threshold}%) - may miss animal",
                video,
            )

    @staticmethod
    def _check_calibration(findings: _Findings, settings: any, is_tst: bool,
                           video: str):
        # TST reports no distances, so calibration is irrelevant to it.
        if is_tst or not hasattr(settings, 'scale_factor'):
            return
        if settings.scale_factor < 0:
            findings.issue("Invalid scale factor (negative)", video)
        elif settings.scale_factor == 0:
            findings.warning(
                "No distance calibration - results will be in pixels", video
            )
        elif getattr(settings.roi_manager, 'is_legacy_bbox_reference', False):
            findings.warning(
                "Calibration came from an ROI bounding box, not a drawn "
                "reference line - redraw the line before trusting cm values",
                video,
            )
        elif getattr(settings.roi_manager, 'reference_is_approximate', False):
            findings.warning(
                "Calibration length was scaled when the template was aligned "
                "(no line endpoints stored) - redraw the reference line "
                "before trusting cm values",
                video,
            )

    @staticmethod
    def _check_time_range(findings: _Findings, settings: any, is_tst: bool,
                          video: str):
        start_time = getattr(settings, 'start_time', None)
        end_time = getattr(settings, 'end_time', None)
        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                findings.issue(
                    f"Invalid time range: end ({end_time}s) "
                    f"<= start ({start_time}s)", video,
                )
        for name, interval in getattr(settings, 'per_roi_times', {}).items():
            if len(interval) != 2 or interval[1] <= interval[0]:
                findings.issue(f"Invalid per-ROI time range for {name}", video)

    @staticmethod
    def _check_roi_definitions(findings: _Findings, settings: any,
                               is_tst: bool, video: str):
        if not hasattr(settings, 'roi_manager'):
            return
        roi_manager = settings.roi_manager
        if len(roi_manager.rois) == 0:
            findings.warning("No ROIs defined (zone-free mode)", video)

        for category, rois in roi_manager.rois.items():
            for roi in rois:
                if len(roi.points) < 3:
                    findings.issue(
                        f"ROI '{category}' has fewer than 3 points", video
                    )

        if roi_manager.source_dimensions is None and roi_manager.rois:
            findings.warning(
                "ROI configuration has no source-video dimensions. "
                "It is a legacy template and should be redrawn before reuse.",
                video,
            )

    # ------------------------------------------------------------------
    # Batch-level checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_mixed_calibration(findings: _Findings,
                                 entries: Sequence[Tuple[any, bool]]):
        """Warn when one batch mixes cm and pixel results."""
        flags = {
            getattr(settings, 'scale_factor', 0.0) > 0
            for settings, is_tst in entries if not is_tst
        }
        if flags == {True, False}:
            findings.warning(
                "Some videos are calibrated (cm) and others are not (pixels); "
                "distances in the batch summary are not comparable"
            )

    @staticmethod
    def _check_output_dir(findings: _Findings, output_dir: Optional[Path]):
        if output_dir is None:
            findings.issue("No output directory selected")
            return
        output_dir = Path(output_dir)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                findings.issue(f"Cannot create output directory: {e}")
                return
        if not os.access(output_dir, os.W_OK):
            findings.issue(f"Output directory not writable: {output_dir}")

    @staticmethod
    def _check_disk_space(findings: _Findings, sizes: Sequence[int],
                          output_dir: Optional[Path]):
        """Check free space once against the summed size of all videos."""
        if output_dir is None:
            return
        try:
            import psutil
        except ImportError:
            logger.debug("psutil not available, skipping disk space check")
            return
        target = Path(output_dir)
        while not target.exists() and target != target.parent:
            target = target.parent
        free_bytes = psutil.disk_usage(str(target)).free
        issue, warning = disk_space_findings(
            estimate_required_bytes(sizes), free_bytes
        )
        if issue:
            findings.issue(issue)
        if warning:
            findings.warning(warning)

    @staticmethod
    def _check_system_resources(findings: _Findings):
        """Check free memory. (A one-shot CPU load reading is only noise.)"""
        try:
            import psutil
        except ImportError:
            logger.debug("psutil not available, skipping system resource check")
            return
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 1:
            findings.issue(
                f"Very low memory: {available_gb:.1f}GB available - "
                "analysis may fail"
            )
        elif available_gb < 2:
            findings.warning(
                f"Low memory: {available_gb:.1f}GB available - "
                "may experience slowdowns"
            )


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
