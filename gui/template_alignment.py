"""
gui/template_alignment.py
Carries a loaded template or batch plan onto videos whose camera moved or
whose resolution differs, using the alignment dialog.

``TemplateAlignment`` is owned by the main window. It remembers one
alignment per video (never changing the template itself) and applies it
wherever per-video settings are built: the run path, the per-video
verification and the batch plan that verification can save.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core import roi_alignment as ra
from core.models import AppConfig, BatchSettings
from gui import ux_helpers
from gui.background_task import TaskCancelled, run_with_progress
from gui.config_checks import (VideoAlignmentCheck, check_videos,
                               format_alignment_checks, probe_dimensions)
from gui.roi_alignment_dialog import align_template_with_dialog
from gui.ux_helpers import Choice

logger = logging.getLogger(__name__)

Dimensions = Tuple[int, int]


@dataclass(frozen=True)
class AlignmentRecord:
    """The alignment chosen for one video, relative to one template."""
    matrix: np.ndarray
    dims: Dimensions
    template_key: str
    rms: Optional[float]
    method: str


def read_middle_frame(video_path: Path) -> Optional[np.ndarray]:
    """The frame the ROI selector shows for a video (its middle one)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 10:
            capture.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok else None


class TemplateAlignment:
    """Per-video alignment of the active template, driven from the main window."""

    def __init__(self, window):
        self.window = window
        self.records: Dict[Path, AlignmentRecord] = {}
        # template fingerprint -> a video known to show the template's view
        self._template_videos: Dict[str, Path] = {}

    # ------------------------------------------------------------ bookkeeping

    def remember_template_video(self, video: Path, manager) -> None:
        """Note which video a template was drawn on (it is its reference view)."""
        self._template_videos[ra.template_key(manager)] = Path(video)

    def record_for(self, video: Path, manager) -> Optional[AlignmentRecord]:
        record = self.records.get(Path(video))
        if record is not None and record.template_key == ra.template_key(manager):
            return record
        return None

    def template_settings_for(self, video: Path) -> Optional[BatchSettings]:
        """The un-aligned configuration the loaded template/plan gives a video."""
        window = self.window
        if window.batch_plan_cache:
            return window._plan_settings_for(video)
        settings = window.current_settings
        if settings is None or settings.roi_manager is None:
            return None
        return settings.copy()

    def apply_records(self, plan: Dict[Path, BatchSettings]) -> int:
        """Replace plan entries that have a stored alignment (in place)."""
        applied = 0
        for video, settings in list(plan.items()):
            record = self.record_for(video, settings.roi_manager)
            if record is None:
                continue
            plan[video] = ra.apply_alignment(settings, record.matrix, record.dims)
            applied += 1
        return applied

    @staticmethod
    def _expected_dims(settings: BatchSettings) -> Optional[Dimensions]:
        dims = settings.roi_manager.source_dimensions
        return tuple(dims) if dims else None

    def mismatched(self, plan: Dict[Path, BatchSettings]) -> List[Tuple[Path, BatchSettings]]:
        """Plan entries whose video frame size differs from their ROIs'."""
        found = []
        for video, settings in plan.items():
            expected = self._expected_dims(settings)
            if expected is None:
                continue
            actual = probe_dimensions(Path(video))
            if actual is not None and tuple(actual) != expected:
                found.append((video, settings))
        return found

    # --------------------------------------------------------- template video

    def _plan_candidate(self, video: Path) -> Optional[Path]:
        """The file a loaded plan's entry for ``video`` was keyed on, if present."""
        window = self.window
        if not window.batch_plan_cache or window.batch_plan_path is None:
            return None
        key = window._match_plan([video]).matched.get(video)
        if key is None:
            return None
        candidate = Path(key)
        if not candidate.is_absolute():
            candidate = Path(window.batch_plan_path).resolve().parent / key
        return candidate if candidate.is_file() else None

    def find_template_video(self, settings: BatchSettings,
                            video: Optional[Path] = None) -> Optional[Path]:
        """A video showing the template's own view of the arena.

        Remembered ones are used when their frame size still matches the
        template's; otherwise the user is asked. None means "give up".
        """
        manager = settings.roi_manager
        expected = manager.source_dimensions
        candidates = [self._template_videos.get(ra.template_key(manager))]
        if video is not None:
            candidates.append(self._plan_candidate(video))
        for candidate in candidates:
            if candidate is not None and Path(candidate).is_file():
                if expected is None or probe_dimensions(Path(candidate)) == tuple(expected):
                    return Path(candidate)
        return self._ask_template_video(manager)

    def _ask_template_video(self, manager) -> Optional[Path]:
        window = self.window
        expected = manager.source_dimensions
        size = f" ({expected[0]}x{expected[1]})" if expected else ""
        filename, _ = QFileDialog.getOpenFileName(
            window, f"Which video was the template drawn on?{size}",
            window.app_settings.dialog_dir("video"),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if not filename:
            return None
        chosen = Path(filename)
        actual = probe_dimensions(chosen)
        if actual is None:
            QMessageBox.warning(
                window, "Cannot Read Video", f"'{chosen.name}' could not be read."
            )
            return None
        if expected is not None and tuple(actual) != tuple(expected):
            QMessageBox.warning(
                window, "Wrong Video Size",
                f"'{chosen.name}' is {actual[0]}x{actual[1]}, but the template "
                f"was drawn on {expected[0]}x{expected[1]} frames.\n\nChoose a "
                "video at the template's frame size that shows the same arena.",
            )
            return None
        self.remember_template_video(chosen, manager)
        return chosen

    # ------------------------------------------------------------------ core

    def _review_one(
        self, video: Path, settings: BatchSettings, template_video: Path,
        registration: Optional["ra.RegistrationResult"],
    ):
        """Open the dialog for one video; its AlignmentChoice or None."""
        template_frame = read_middle_frame(template_video)
        target_frame = read_middle_frame(video)
        if template_frame is None or target_frame is None:
            QMessageBox.warning(
                self.window, "Cannot Read Video",
                f"A frame could not be read from '{template_video.name}' or "
                f"'{Path(video).name}'.",
            )
            return None
        samples = AppConfig().video.ref_frame_samples

        def auto_align(should_cancel):
            return ra.register_videos(
                template_video, video, samples, should_cancel
            )

        return align_template_with_dialog(
            self.window, settings.roi_manager, template_frame, target_frame,
            auto_align, template_name=template_video.name,
            target_name=Path(video).name, initial_registration=registration,
        )

    def align_videos(
        self,
        items: List[Tuple[Path, BatchSettings]],
        registrations: Optional[Dict[Path, "ra.RegistrationResult"]] = None,
    ) -> Optional[Dict[Path, BatchSettings]]:
        """Interactively align each (video, template settings) pair.

        After a video is aligned the user may apply the same transform to the
        remaining videos that share its frame size (one camera). Returns the
        aligned settings per video, or None if the user cancelled anywhere
        (alignments finished before that are still remembered).
        """
        registrations = registrations or {}
        pending = list(items)
        done: Dict[Path, BatchSettings] = {}
        while pending:
            video, settings = pending.pop(0)
            template_video = self.find_template_video(settings, video)
            if template_video is None:
                return None
            choice = self._review_one(
                video, settings, template_video, registrations.get(video)
            )
            if choice is None:
                return None
            dims = probe_dimensions(Path(video))
            done[video] = self._store(video, settings, choice, dims)

            key = ra.template_key(settings.roi_manager)
            same_camera = [
                (other, other_settings) for other, other_settings in pending
                if probe_dimensions(Path(other)) == dims
                and ra.template_key(other_settings.roi_manager) == key
            ]
            if same_camera and self._ask_apply_to_same_camera(len(same_camera), dims):
                for other, other_settings in same_camera:
                    done[other] = self._store(other, other_settings, choice, dims)
                pending = [p for p in pending if p not in same_camera]
        return done

    def _store(self, video: Path, settings: BatchSettings, choice, dims) -> BatchSettings:
        self.records[Path(video)] = AlignmentRecord(
            choice.matrix, tuple(dims), ra.template_key(settings.roi_manager),
            choice.rms, choice.method,
        )
        logger.info(
            "Aligned template to %s (%s, residual %s)", Path(video).name,
            choice.method,
            "n/a" if choice.rms is None else f"{choice.rms:.2f} px",
        )
        return ra.apply_alignment(settings, choice.matrix, dims)

    def _ask_apply_to_same_camera(self, count: int, dims: Dimensions) -> bool:
        answer = ux_helpers.ask_choice(
            self.window, "Same Camera?",
            f"{count} more video{'s' if count != 1 else ''} in the list "
            f"{'have' if count != 1 else 'has'} the same frame size "
            f"({dims[0]}x{dims[1]}).",
            [
                Choice("all", "Use this alignment for all of them"),
                Choice("each", "Align each one separately",
                       QMessageBox.ButtonRole.ActionRole),
            ],
            default_key="all",
            informative=(
                "Choose the first only if they were recorded with the camera "
                "in the same place."
            ),
        )
        return answer == "all"

    # ----------------------------------------------------------- entry points

    def prepare_plan(self, plan: Dict[Path, BatchSettings]) -> bool:
        """Before a run: apply stored alignments, align size mismatches.

        Mutates ``plan`` in place. False means the run must not start (the
        user declined or cancelled alignment for a video that still does not
        fit its ROIs); the pre-flight check would block it anyway.
        """
        self.apply_records(plan)
        mismatched = self.mismatched(plan)
        if not mismatched:
            return True
        names = "\n".join(
            f"  - {Path(video).name}" for video, _ in mismatched[:10]
        )
        if len(mismatched) > 10:
            names += f"\n  ... and {len(mismatched) - 10} more"
        answer = ux_helpers.ask_choice(
            self.window, "Align Template?",
            f"{len(mismatched)} video(s) have a different frame size than the "
            f"template's ROIs, so the regions would be misplaced:\n\n{names}",
            [
                Choice("align", "Align template..."),
                Choice("cancel", "Cancel", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="align",
            informative=(
                "Aligning carries the ROIs (and the calibration) over to "
                "each video; you review the result before anything runs."
            ),
        )
        if answer != "align":
            return False
        aligned = self.align_videos(mismatched)
        if aligned is None:
            return False
        for video, settings in aligned.items():
            plan[video] = settings
        return not self.mismatched(plan)

    def settings_for_review(
        self, video: Path, base_settings: BatchSettings,
        current_settings: BatchSettings,
    ) -> BatchSettings:
        """Settings to show when verifying ``video``.

        A stored alignment is applied; otherwise a frame-size mismatch with
        the settings carried over from the previous video starts the
        alignment. Anything else is returned unchanged.
        """
        record = self.record_for(video, base_settings.roi_manager)
        if record is not None:
            return ra.apply_alignment(
                current_settings, record.matrix, record.dims,
                base_settings.roi_manager,
            )
        expected = self._expected_dims(current_settings)
        actual = probe_dimensions(Path(video))
        if expected is not None and actual is not None and tuple(actual) != expected:
            aligned = self.align_one(video, base_settings, current_settings)
            if aligned is not None:
                return aligned
        return current_settings

    def align_one(
        self, video: Path, base_settings: BatchSettings,
        current_settings: BatchSettings,
    ) -> Optional[BatchSettings]:
        """Align the ORIGINAL template to ``video`` (never a chained result).

        ``current_settings`` supplies the times and calibration the user has
        set so far; they are kept. None if the user cancelled.
        """
        template_settings = ra.with_roi_manager(
            current_settings, base_settings.roi_manager
        )
        aligned = self.align_videos([(video, template_settings)])
        if aligned is None:
            return None
        return aligned.get(video)

    def check_all(self) -> None:
        """Tools action: flag videos that do not fit the template, then align."""
        window = self.window
        videos = [Path(v) for v in window.video_paths]
        if not videos:
            QMessageBox.information(
                window, "Align Template", "Add one or more videos first."
            )
            return
        groups: Dict[str, List[Tuple[Path, BatchSettings]]] = {}
        for video in videos:
            settings = self.template_settings_for(video)
            if settings is not None:
                groups.setdefault(ra.template_key(settings.roi_manager), []).append(
                    (video, settings)
                )
        if not groups:
            QMessageBox.information(
                window, "Align Template",
                "Load a template or batch plan first (File > Load "
                "Configuration / Plan...), or finish a setup.",
            )
            return

        checks: Dict[Path, VideoAlignmentCheck] = {}
        samples = AppConfig().video.ref_frame_samples
        for items in groups.values():
            template_settings = items[0][1]
            template_video = self.find_template_video(template_settings, items[0][0])
            if template_video is None:
                return
            group_videos = [video for video, _ in items]
            try:
                results = run_with_progress(
                    window, "Comparing each video's background with the template...",
                    lambda cancel, tv=template_video, gv=group_videos,
                    manager=template_settings.roi_manager: check_videos(
                        manager, tv, gv, samples, cancel
                    ),
                    title="Checking Alignment",
                )
            except TaskCancelled:
                return
            except Exception as exc:
                logger.warning("Alignment check failed", exc_info=True)
                QMessageBox.warning(
                    window, "Alignment Check Failed",
                    f"The videos could not be compared:\n{exc}",
                )
                return
            checks.update({check.video: check for check in results})

        flagged = [c for c in checks.values() if c.needs_alignment]
        summary = format_alignment_checks(list(checks.values()))
        if not flagged:
            QMessageBox.information(window, "Template Alignment", summary)
            return
        answer = ux_helpers.ask_choice(
            window, "Template Alignment", summary,
            [
                Choice("review", f"Review {len(flagged)} video(s)..."),
                Choice("close", "Close", QMessageBox.ButtonRole.RejectRole),
            ],
            default_key="review", icon=QMessageBox.Icon.Warning,
        )
        if answer != "review":
            return
        items = []
        for check in flagged:
            settings = self.template_settings_for(check.video)
            if settings is not None:
                items.append((check.video, settings))
        registrations = {c.video: c.registration for c in flagged if c.registration}
        aligned = self.align_videos(items, registrations)
        if aligned:
            QMessageBox.information(
                window, "Template Alignment",
                f"Aligned {len(aligned)} video(s). The aligned regions are "
                "used when the analysis starts and are saved with a batch plan.",
            )
