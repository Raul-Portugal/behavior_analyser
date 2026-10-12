"""
gui/config_checks.py
Checks that can run the moment a template or plan is chosen, before any
dialog asks the user to configure anything.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cv2

from core import roi_alignment

Dimensions = Tuple[int, int]

# ROIs that would sit further than this (RMS, px) from where the video's
# own background says they belong are flagged for an alignment review.
ALIGNMENT_FLAG_PX = 3.0

STATUS_OK = "ok"
STATUS_SIZE_DIFFERS = "size_differs"
STATUS_SHIFTED = "shifted"
STATUS_UNVERIFIED = "unverified"
STATUS_UNREADABLE = "unreadable"


@dataclass(frozen=True)
class DimensionMismatch:
    video: Path
    expected: Dimensions
    actual: Optional[Dimensions]  # None: the file could not be read


def probe_dimensions(path: Path) -> Optional[Dimensions]:
    """Frame size of a video without decoding it, or None if unreadable."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return (width, height) if width > 0 and height > 0 else None


def dimension_mismatches(
    videos: Iterable[Path],
    expected_for: Callable[[Path], Optional[Dimensions]],
    prober: Callable[[Path], Optional[Dimensions]] = probe_dimensions,
) -> List[DimensionMismatch]:
    """Videos whose frame size differs from the size their ROIs were drawn on.

    ``expected_for`` returns the size recorded with a video's ROIs, or None
    when nothing is recorded (legacy settings), in which case it is skipped.
    """
    found: List[DimensionMismatch] = []
    for video in videos:
        expected = expected_for(video)
        if expected is None:
            continue
        actual = prober(video)
        if actual is None or tuple(actual) != tuple(expected):
            found.append(DimensionMismatch(
                video, (int(expected[0]), int(expected[1])), actual
            ))
    return found


def format_dimension_mismatches(
    mismatches: List[DimensionMismatch], limit: int = 8
) -> str:
    """Human-readable list for a message box (empty string if none)."""
    if not mismatches:
        return ""
    lines = []
    for item in mismatches[:limit]:
        want = f"{item.expected[0]}x{item.expected[1]}"
        have = (
            f"{item.actual[0]}x{item.actual[1]}"
            if item.actual else "could not be read"
        )
        lines.append(f"  - {item.video.name}: ROIs drawn for {want}, video is {have}")
    if len(mismatches) > limit:
        lines.append(f"  ... and {len(mismatches) - limit} more")
    count = len(mismatches)
    return (
        f"{count} video{'s' if count != 1 else ''} in the list "
        f"{'do' if count != 1 else 'does'} not match the frame size the "
        "loaded ROIs were drawn on:\n" + "\n".join(lines) +
        "\n\nRegions would be misplaced. Use Tools > Align template to "
        "videos... to carry the regions over, remove those videos, or choose "
        "'Set up from scratch' when starting."
    )


@dataclass
class VideoAlignmentCheck:
    """How well one video agrees with the template's view of the arena."""
    video: Path
    dims: Optional[Dimensions]
    status: str
    registration: Optional["roi_alignment.RegistrationResult"] = None
    displacement_px: Optional[float] = None
    note: str = ""

    @property
    def needs_alignment(self) -> bool:
        return self.status in (STATUS_SIZE_DIFFERS, STATUS_SHIFTED)


def check_videos(
    template_manager,
    template_video: Path,
    videos: Iterable[Path],
    num_samples: int = 100,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    threshold_px: float = ALIGNMENT_FLAG_PX,
    prober: Callable[[Path], Optional[Dimensions]] = probe_dimensions,
    generate=None,
) -> List[VideoAlignmentCheck]:
    """Compare every video's background with the template video's.

    Each video's animal-free reference frame (cached on disk, and reused by
    the analysis itself) is registered against the template's. A video is
    flagged when its frame size differs, or when the template's ROIs would be
    displaced by more than ``threshold_px`` (RMS) because the camera moved.
    Videos whose arena cannot be matched are reported as unverified, never
    flagged. ``InterruptedError`` propagates when ``should_cancel`` fires.
    """
    template_dims = template_manager.source_dimensions
    video_list = list(videos)

    if generate is None:
        from core.video import ReferenceFrameGenerator

        def generate(path, samples, cancel):
            return ReferenceFrameGenerator.generate(
                path, samples, use_cache=True, should_cancel=cancel
            )

    template_frame = generate(template_video, num_samples, should_cancel)
    results: List[VideoAlignmentCheck] = []
    for index, video in enumerate(video_list):
        if progress is not None:
            progress(index, len(video_list), Path(video).name)
        if should_cancel is not None and should_cancel():
            raise InterruptedError()
        dims = prober(video)
        if dims is None:
            results.append(VideoAlignmentCheck(
                video, None, STATUS_UNREADABLE, note="the video could not be read"
            ))
            continue
        try:
            frame = generate(video, num_samples, should_cancel)
            registration = roi_alignment.auto_register(template_frame, frame)
        except InterruptedError:
            raise
        except Exception as exc:
            results.append(VideoAlignmentCheck(
                video, dims, STATUS_UNVERIFIED,
                note=f"could not build a background frame ({exc})",
            ))
            continue

        size_differs = (
            template_dims is not None and tuple(dims) != tuple(template_dims)
        )
        if size_differs:
            results.append(VideoAlignmentCheck(
                video, dims, STATUS_SIZE_DIFFERS,
                registration if registration.usable else None,
                note=f"frame size {dims[0]}x{dims[1]} differs from the "
                     f"template's {template_dims[0]}x{template_dims[1]}",
            ))
        elif not registration.usable or registration.verdict == roi_alignment.VERDICT_POOR:
            results.append(VideoAlignmentCheck(
                video, dims, STATUS_UNVERIFIED, registration,
                note="the arena could not be matched reliably "
                     "(too little texture in the background?)",
            ))
        else:
            shift = roi_alignment.roi_displacement_px(
                template_manager, registration.matrix
            )
            flagged = shift > threshold_px
            results.append(VideoAlignmentCheck(
                video, dims, STATUS_SHIFTED if flagged else STATUS_OK,
                registration, shift,
                note=(f"the camera moved: the ROIs would be off by "
                      f"{shift:.1f} px" if flagged else f"{shift:.1f} px"),
            ))
    if progress is not None:
        progress(len(video_list), len(video_list), "")
    return results


def format_alignment_checks(checks: List[VideoAlignmentCheck], limit: int = 12) -> str:
    """Human-readable summary of ``check_videos`` results."""
    flagged = [c for c in checks if c.needs_alignment]
    unverified = [c for c in checks if c.status in (STATUS_UNVERIFIED, STATUS_UNREADABLE)]
    ok = len(checks) - len(flagged) - len(unverified)
    lines = [f"{ok} of {len(checks)} video(s) match the template."]
    if flagged:
        lines.append(f"\n{len(flagged)} need alignment:")
        lines += [f"  - {c.video.name}: {c.note}" for c in flagged[:limit]]
        if len(flagged) > limit:
            lines.append(f"  ... and {len(flagged) - limit} more")
    if unverified:
        lines.append(f"\n{len(unverified)} could not be checked:")
        lines += [f"  - {c.video.name}: {c.note}" for c in unverified[:limit]]
    return "\n".join(lines)
