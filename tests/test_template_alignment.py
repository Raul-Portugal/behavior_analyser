"""Applying a template to videos from another camera placement or resolution."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core import roi_alignment as ra
from core.models import AppConfig, BatchSettings
from core.video import ReferenceFrameGenerator
from gui.config_checks import (STATUS_OK, STATUS_SHIFTED, STATUS_SIZE_DIFFERS,
                               STATUS_UNVERIFIED, check_videos,
                               format_alignment_checks, format_dimension_mismatches,
                               DimensionMismatch)
from preflight_checker import PreflightChecker, VideoProbe
from roi_manager import ROIManager
from tests.alignment_support import (TEMPLATE_DIMS, arena_image, camera_move,
                                     max_grid_error, rescale, warp)

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui import ux_helpers
    from gui.main_window import MainWindow
    from gui.roi_alignment_dialog import RoiAlignmentDialog
    from gui.settings_dialog import SettingsDialog
    from mazes import create_maze
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

SHIFT_TRUTH = camera_move(shift=(24.0, -18.0), rotation_deg=2.0)
HALF_TRUTH = rescale(camera_move(shift=(6.0, -4.0)), 0.5)


def _write_video(path: Path, image: np.ndarray, frames: int = 12) -> Path:
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
    )
    for _ in range(frames):
        writer.write(image)
    writer.release()
    return path


def _template_settings(scale=0.05, with_line=True) -> BatchSettings:
    manager = ROIManager(source_dimensions=TEMPLATE_DIMS)
    manager.add_roi("arm_a", np.array([[280, 60], [360, 60], [360, 200], [280, 200]]))
    manager.add_roi("arm_b", np.array([[80, 200], [260, 200], [260, 280], [80, 280]]))
    manager.add_roi("arm_c", np.array([[380, 200], [560, 200], [560, 280], [380, 280]]))
    manager.add_roi("center", np.array([[270, 200], [370, 200], [370, 280], [270, 280]]))
    if with_line:
        manager.set_reference_line((80, 240), (260, 240), "User-drawn reference line")
    else:
        manager.set_reference_length(180.0, "User-drawn reference line")
    return BatchSettings(manager, AppConfig().detection, scale)


class _VideoFixture(unittest.TestCase):
    """Synthetic videos recorded by 'cameras' placed differently."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls._tmp.name)
        scene = arena_image()
        cls.template_video = _write_video(root / "template.avi", scene)
        cls.same_video = _write_video(root / "same.avi", scene)
        cls.same_video_2 = _write_video(root / "same2.avi", scene)
        cls.shifted_video = _write_video(
            root / "shifted.avi", warp(scene, SHIFT_TRUTH, TEMPLATE_DIMS)
        )
        cls.half_video = _write_video(
            root / "half.avi", warp(scene, HALF_TRUTH, (320, 240))
        )
        cls.half_video_2 = _write_video(
            root / "half2.avi", warp(scene, HALF_TRUTH, (320, 240))
        )
        cls.blank_video = _write_video(
            root / "blank.avi", np.full((480, 640, 3), 128, dtype=np.uint8)
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        cache = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(cache.cleanup)
        patcher = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=Path(cache.name),
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class CheckVideosTests(_VideoFixture):
    def _check(self, *videos, settings=None):
        settings = settings or _template_settings()
        return check_videos(
            settings.roi_manager, self.template_video, list(videos), num_samples=6
        )

    def test_identical_camera_is_not_flagged(self):
        (check,) = self._check(self.same_video)
        self.assertEqual(check.status, STATUS_OK)
        self.assertFalse(check.needs_alignment)
        self.assertLess(check.displacement_px, 1.0)

    def test_shifted_camera_is_flagged(self):
        (check,) = self._check(self.shifted_video)
        self.assertEqual(check.status, STATUS_SHIFTED)
        self.assertTrue(check.needs_alignment)
        self.assertGreater(check.displacement_px, 10.0)
        self.assertLess(
            max_grid_error(check.registration.matrix, SHIFT_TRUTH), 1.0
        )
        self.assertIn("camera moved", check.note)

    def test_different_resolution_is_flagged_with_a_usable_suggestion(self):
        (check,) = self._check(self.half_video)
        self.assertEqual(check.status, STATUS_SIZE_DIFFERS)
        self.assertEqual(check.dims, (320, 240))
        self.assertLess(max_grid_error(check.registration.matrix, HALF_TRUTH), 1.0)

    def test_untextured_video_is_unverified_not_flagged(self):
        (check,) = self._check(self.blank_video)
        self.assertEqual(check.status, STATUS_UNVERIFIED)
        self.assertFalse(check.needs_alignment)

    def test_unreadable_video_is_reported(self):
        (check,) = self._check(self.same_video.parent / "missing.avi")
        self.assertFalse(check.needs_alignment)
        self.assertIn("could not be read", check.note)

    def test_summary_lists_flagged_videos(self):
        checks = self._check(self.same_video, self.shifted_video, self.half_video)
        text = format_alignment_checks(checks)
        self.assertIn("1 of 3 video(s) match", text)
        self.assertIn("shifted.avi", text)
        self.assertIn("half.avi", text)

    def test_cancel_is_propagated(self):
        with self.assertRaises(InterruptedError):
            check_videos(
                _template_settings().roi_manager, self.template_video,
                [self.same_video], num_samples=6, should_cancel=lambda: True,
            )

    def test_mismatch_message_points_at_alignment(self):
        text = format_dimension_mismatches(
            [DimensionMismatch(Path("a.mp4"), (160, 120), (640, 480))]
        )
        self.assertIn("Align template to videos", text)


class PreflightGuidanceTests(unittest.TestCase):
    def _issues(self, settings, width, height):
        def prober(path):
            return VideoProbe(
                path=path, size_bytes=10 ** 6, opened=True, width=width,
                height=height, fps=30.0, frame_count=300,
                mean_brightness=100.0, measured_duration_s=None,
            )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                patch("psutil.virtual_memory",
                      return_value=SimpleNamespace(available=16 * 1024 ** 3)), \
                patch("psutil.disk_usage",
                      return_value=SimpleNamespace(free=1000 * 1024 ** 3)):
            return PreflightChecker().validate_batch(
                {Path("a.mp4"): settings}, Path(tmp), prober=prober
            )

    def test_mismatch_is_still_blocking_and_points_to_alignment(self):
        issues, _ = self._issues(_template_settings(), 320, 240)
        geometry = [i for i in issues if "ROI geometry" in i]
        self.assertEqual(len(geometry), 1)
        self.assertIn("Align the template", geometry[0])

    def test_aligned_settings_pass_the_dimension_check(self):
        aligned = ra.apply_alignment(_template_settings(), HALF_TRUTH, (320, 240))
        issues, _ = self._issues(aligned, 320, 240)
        self.assertFalse([i for i in issues if "ROI geometry" in i])

    def test_length_only_alignment_warns_about_approximate_calibration(self):
        aligned = ra.apply_alignment(
            _template_settings(with_line=False), HALF_TRUTH, (320, 240)
        )
        _, warnings = self._issues(aligned, 320, 240)
        self.assertTrue(any("scaled when the template was aligned" in w for w in warnings))


def _accept_with_auto_align(dialog):
    dialog.run_auto_align()
    return 1


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class MainWindowAlignmentTests(_VideoFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        self.template = _template_settings()
        self.window.current_settings = self.template
        self.window._alignment.remember_template_video(
            self.template_video, self.template.roi_manager
        )

    def _plan(self, *videos):
        self.window.video_paths = list(videos)
        return {video: self.template.copy() for video in videos}

    def _prepare(self, plan, answers=("align",)):
        with patch.object(ux_helpers, "ask_choice", side_effect=list(answers)) as ask, \
                patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align):
            return self.window._alignment.prepare_plan(plan), ask

    def test_identical_dimension_videos_are_left_alone(self):
        plan = self._plan(self.same_video, self.same_video_2)
        before = {v: s.roi_manager.to_dict() for v, s in plan.items()}
        ok, ask = self._prepare(plan)
        self.assertTrue(ok)
        ask.assert_not_called()
        self.assertEqual(
            {v: s.roi_manager.to_dict() for v, s in plan.items()}, before
        )

    def test_different_resolution_goes_through_alignment(self):
        plan = self._plan(self.half_video)
        ok, ask = self._prepare(plan)
        self.assertTrue(ok)
        self.assertEqual(ask.call_args_list[0].args[1], "Align Template?")
        aligned = plan[self.half_video]
        manager = aligned.roi_manager
        self.assertEqual(manager.source_dimensions, (320, 240))
        expected = ra.transform_roi_manager(
            self.template.roi_manager, HALF_TRUTH, (320, 240)
        )
        for category in expected.rois:
            diff = manager.rois[category][0].points - expected.rois[category][0].points
            self.assertLessEqual(int(np.max(np.abs(diff))), 1, category)
        # The 180 px line is ~90 px now, and 0.05 cm/px became ~0.1 cm/px.
        self.assertAlmostEqual(manager.reference_length_pixels, 90.0, delta=1.0)
        self.assertAlmostEqual(aligned.scale_factor, 0.1, delta=0.005)
        self.assertAlmostEqual(
            aligned.scale_factor * manager.reference_length_pixels, 9.0, delta=0.05
        )
        # The template itself is never modified.
        self.assertEqual(self.template.roi_manager.source_dimensions, TEMPLATE_DIMS)
        self.assertEqual(self.window._alignment.mismatched(plan), [])

    def test_aligned_plan_passes_preflight_dimension_check(self):
        plan = self._plan(self.half_video)
        self._prepare(plan)
        issues, _ = PreflightGuidanceTests()._issues(plan[self.half_video], 320, 240)
        self.assertFalse([i for i in issues if "ROI geometry" in i])

    def test_one_alignment_can_serve_every_video_from_the_same_camera(self):
        plan = self._plan(self.half_video, self.same_video, self.half_video_2)
        exec_calls = []

        def counting(dialog):
            exec_calls.append(1)
            return _accept_with_auto_align(dialog)

        with patch.object(ux_helpers, "ask_choice", side_effect=["align", "all"]) as ask,                 patch.object(RoiAlignmentDialog, "exec", counting):
            ok = self.window._alignment.prepare_plan(plan)
        self.assertTrue(ok)
        self.assertEqual(len(exec_calls), 1)
        self.assertEqual(ask.call_args_list[1].args[1], "Same Camera?")
        first = plan[self.half_video].roi_manager
        second = plan[self.half_video_2].roi_manager
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(second.source_dimensions, (320, 240))
        # The same-size video was never part of it.
        self.assertEqual(
            plan[self.same_video].roi_manager.source_dimensions, TEMPLATE_DIMS
        )

    def test_each_video_can_be_aligned_separately(self):
        plan = self._plan(self.half_video, self.half_video_2)
        exec_calls = []

        def counting(dialog):
            exec_calls.append(1)
            return _accept_with_auto_align(dialog)

        with patch.object(ux_helpers, "ask_choice", side_effect=["align", "each"]), \
                patch.object(RoiAlignmentDialog, "exec", counting):
            self.assertTrue(self.window._alignment.prepare_plan(plan))
        self.assertEqual(len(exec_calls), 2)

    def test_declining_alignment_stops_the_run_untouched(self):
        plan = self._plan(self.half_video)
        before = plan[self.half_video].roi_manager.to_dict()
        ok, _ = self._prepare(plan, answers=("cancel",))
        self.assertFalse(ok)
        self.assertEqual(plan[self.half_video].roi_manager.to_dict(), before)

    def test_cancelling_the_dialog_stops_the_run(self):
        plan = self._plan(self.half_video)
        with patch.object(ux_helpers, "ask_choice", return_value="align"), \
                patch.object(RoiAlignmentDialog, "exec", lambda dialog: 0):
            self.assertFalse(self.window._alignment.prepare_plan(plan))

    def test_validate_analysis_plan_aligns_before_preflight(self):
        plan = self._plan(self.half_video)
        seen = {}

        def fake_preflight(analysis_plan):
            seen["dims"] = analysis_plan[self.half_video].roi_manager.source_dimensions
            return [], []

        with patch.object(self.window, "_run_preflight", side_effect=fake_preflight), \
                patch.object(ux_helpers, "ask_choice", side_effect=["align", "start"]), \
                patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align), \
                patch.object(QMessageBox, "critical") as critical:
            result = self.window._validate_analysis_plan(plan)
            self.assertTrue(result, critical.call_args)
        self.assertEqual(seen["dims"], (320, 240))

    def test_stored_alignment_is_applied_on_a_later_run(self):
        self.window.video_paths = [self.shifted_video]
        # No size mismatch, so nothing is asked; the user aligns explicitly.
        with patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align):
            aligned = self.window._alignment.align_videos(
                [(self.shifted_video, self.template.copy())]
            )
        self.assertLess(
            max_grid_error(
                self.window._alignment.records[self.shifted_video].matrix,
                SHIFT_TRUTH,
            ), 1.0,
        )
        later = self._plan(self.shifted_video)
        with patch.object(ux_helpers, "ask_choice") as ask:
            self.assertTrue(self.window._alignment.prepare_plan(later))
        ask.assert_not_called()
        self.assertEqual(
            later[self.shifted_video].roi_manager.to_dict(),
            aligned[self.shifted_video].roi_manager.to_dict(),
        )
        self.assertNotEqual(
            later[self.shifted_video].roi_manager.to_dict(),
            self.template.roi_manager.to_dict(),
        )

    def test_records_of_another_template_are_ignored(self):
        with patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align):
            self.window._alignment.align_videos(
                [(self.shifted_video, self.template.copy())]
            )
        other = _template_settings()
        other.roi_manager.add_roi("arm_d", np.array([[10, 10], [60, 10], [60, 60]]))
        plan = {self.shifted_video: other}
        before = other.roi_manager.to_dict()
        self.assertTrue(self.window._alignment.prepare_plan(plan))
        self.assertEqual(plan[self.shifted_video].roi_manager.to_dict(), before)

    def test_tools_check_flags_the_shifted_video_and_aligns_it(self):
        self.window.video_paths = [self.same_video, self.shifted_video]
        with patch.object(ux_helpers, "ask_choice", return_value="review") as ask, \
                patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align), \
                patch.object(QMessageBox, "information") as info:
            self.window._alignment.check_all()
        summary = ask.call_args.args[2]
        self.assertIn("shifted.avi", summary)
        self.assertNotIn("same.avi:", summary)
        records = self.window._alignment.records
        self.assertIn(self.shifted_video, records)
        self.assertNotIn(self.same_video, records)
        info.assert_called_once()

    def test_tools_check_with_nothing_flagged_only_informs(self):
        self.window.video_paths = [self.same_video, self.same_video_2]
        with patch.object(ux_helpers, "ask_choice") as ask, \
                patch.object(QMessageBox, "information") as info:
            self.window._alignment.check_all()
        ask.assert_not_called()
        self.assertIn("2 of 2", info.call_args.args[2])

    def test_tools_check_needs_a_template(self):
        self.window.current_settings = None
        self.window.video_paths = [self.same_video]
        with patch.object(QMessageBox, "information") as info:
            self.window._alignment.check_all()
        self.assertIn("Load a template", info.call_args.args[2])

    def test_verification_starts_alignment_for_a_size_mismatch(self):
        with patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align):
            result = self.window._alignment.settings_for_review(
                self.half_video, self.template, self.template.copy()
            )
        self.assertEqual(result.roi_manager.source_dimensions, (320, 240))

    def test_verification_keeps_matching_settings_unchanged(self):
        current = self.template.copy()
        with patch.object(RoiAlignmentDialog, "exec") as dialog_exec:
            result = self.window._alignment.settings_for_review(
                self.same_video, self.template, current
            )
        dialog_exec.assert_not_called()
        self.assertIs(result, current)

    def test_check_alignment_in_verification_realigns_the_original_template(self):
        # The chained settings were already aligned to another video; the
        # new alignment must start again from the template, keeping times.
        chained = ra.apply_alignment(self.template, HALF_TRUTH, (320, 240))
        chained.start_time = 1.5
        with patch.object(RoiAlignmentDialog, "exec", _accept_with_auto_align):
            result = self.window._alignment.align_one(
                self.shifted_video, self.template, chained
            )
        self.assertEqual(result.start_time, 1.5)
        self.assertEqual(result.roi_manager.source_dimensions, TEMPLATE_DIMS)
        expected = ra.transform_roi_manager(
            self.template.roi_manager, SHIFT_TRUTH, TEMPLATE_DIMS
        )
        diff = (result.roi_manager.rois["arm_a"][0].points
                - expected.rois["arm_a"][0].points)
        self.assertLessEqual(int(np.max(np.abs(diff))), 1)
        # Calibration is relative to the template again, not the 0.1 chain.
        self.assertAlmostEqual(result.scale_factor, 0.05, delta=0.003)

    def test_verification_dialog_offers_check_alignment(self):
        dialog = SettingsDialog(
            str(self.same_video), self.template.copy(), is_verification=True,
            maze_name="Y-Maze",
        )
        self.addCleanup(dialog.deleteLater)
        dialog.align_btn.click()
        self.assertEqual(dialog.result_action, "align")

    def test_save_plan_keeps_aligned_dimensions(self):
        plan = self._plan(self.half_video)
        self._prepare(plan)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            from core.batch_plan_keys import build_plan_by_key
            path = Path(tmp) / "plan.json"
            BatchSettings.save_batch_plan(build_plan_by_key(plan, path), path)
            loaded = BatchSettings.load_batch_plan(path)
        (settings,) = loaded.values()
        self.assertEqual(settings.roi_manager.source_dimensions, (320, 240))
        self.assertIsNotNone(settings.roi_manager.reference_line)


if __name__ == "__main__":
    unittest.main()
