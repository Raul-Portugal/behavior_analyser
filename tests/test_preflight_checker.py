import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.models import AppConfig, BatchSettings
from preflight_checker import (PreflightChecker, VideoProbe,
                               disk_space_findings, estimate_required_bytes,
                               probe_video, vfr_warning)
from roi_manager import ROIManager

GB = 1024 ** 3


def _settings(paradigm_id=None, scale=0.0, **kwargs) -> BatchSettings:
    manager = ROIManager((32, 32))
    manager.add_roi("arm_a", np.array([[0, 0], [30, 0], [30, 30], [0, 30]]))
    return BatchSettings(
        manager, AppConfig().detection, scale,
        paradigm_id=paradigm_id, **kwargs
    )


def _prober(size=GB, fps=30.0, frames=300, measured=None, **overrides):
    def probe(path: Path) -> VideoProbe:
        values = dict(
            path=path, size_bytes=size, opened=True, width=32, height=32,
            fps=fps, frame_count=frames, mean_brightness=100.0,
            measured_duration_s=measured,
        )
        values.update(overrides)
        return VideoProbe(**values)
    return probe


class BatchPreflightTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        # Plenty of memory by default; disk is patched per test.
        memory = patch("psutil.virtual_memory",
                       return_value=SimpleNamespace(available=16 * GB))
        memory.start()
        self.addCleanup(memory.stop)

    def _run(self, plan, free_bytes=1000 * GB, **kwargs):
        with patch("psutil.disk_usage",
                   return_value=SimpleNamespace(free=free_bytes)):
            return PreflightChecker().validate_batch(plan, self.out, **kwargs)

    def test_disk_requirement_is_summed_over_the_batch(self):
        self.assertAlmostEqual(
            estimate_required_bytes([GB, 3 * GB]), 4 * GB * 0.15
        )
        plan = {Path(f"v{i}.mp4"): _settings(scale=0.1) for i in range(4)}
        # 4 x 1 GB videos need ~0.6 GB; one video alone would need 0.15 GB.
        issues, _ = self._run(plan, free_bytes=0.4 * GB, prober=_prober())
        self.assertTrue(any("Insufficient disk space" in i for i in issues))
        single = {Path("v0.mp4"): _settings(scale=0.1)}
        issues, _ = self._run(single, free_bytes=0.4 * GB, prober=_prober())
        self.assertFalse(any("disk space" in i for i in issues))

    def test_disk_findings_levels(self):
        issue, warning = disk_space_findings(10 * GB, 5 * GB)
        self.assertIsNotNone(issue)
        issue, warning = disk_space_findings(10 * GB, 15 * GB)
        self.assertIsNone(issue)
        self.assertIsNotNone(warning)
        self.assertEqual(disk_space_findings(10 * GB, 50 * GB), (None, None))

    def test_global_checks_run_once_not_per_video(self):
        plan = {Path(f"v{i}.mp4"): _settings(scale=0.1) for i in range(3)}
        with patch("psutil.disk_usage",
                   return_value=SimpleNamespace(free=100 * GB)) as disk:
            PreflightChecker().validate_batch(plan, self.out, prober=_prober())
        self.assertEqual(disk.call_count, 1)

    def test_no_calibration_warning_skips_tst(self):
        plain = {Path("a.mp4"): _settings(paradigm_id="Y-Maze")}
        _, warnings = self._run(plain, prober=_prober())
        self.assertTrue(any("No distance calibration" in w for w in warnings))

        tst = {Path("a.mp4"): _settings(paradigm_id="Tail Suspension Test")}
        _, warnings = self._run(tst, prober=_prober())
        self.assertFalse(any("calibration" in w for w in warnings))

        legacy = {Path("a.mp4"): _settings()}
        _, warnings = self._run(
            legacy, prober=_prober(), paradigm_name="Tail Suspension Test"
        )
        self.assertFalse(any("calibration" in w for w in warnings))

    def test_identical_warnings_are_merged_across_videos(self):
        plan = {Path(f"v{i}.mp4"): _settings(paradigm_id="Y-Maze")
                for i in range(3)}
        _, warnings = self._run(plan, prober=_prober())
        matching = [w for w in warnings if "No distance calibration" in w]
        self.assertEqual(len(matching), 1)
        self.assertIn("affects 3 videos", matching[0])

    def test_one_failing_check_does_not_skip_the_rest(self):
        class Exploding(SimpleNamespace):
            @property
            def detection_config(self):
                raise RuntimeError("boom")

        base = _settings(scale=0.1)
        broken = Exploding(
            roi_manager=base.roi_manager, scale_factor=0.1,
            start_time=10.0, end_time=5.0, per_roi_times={},
            paradigm_id=None,
        )
        # The detection check raises; only that check may be lost.
        issues, _ = self._run({Path("a.mp4"): broken}, prober=_prober())
        self.assertTrue(any("detection settings" in i and "boom" in i for i in issues))
        self.assertTrue(any("Invalid time range" in i for i in issues))

    def test_prober_crash_is_a_blocking_issue(self):
        def crashing(_path):
            raise OSError("disk unplugged")
        plan = {Path("a.mp4"): _settings(scale=0.1)}
        issues, _ = self._run(plan, prober=crashing)
        self.assertTrue(any("disk unplugged" in i for i in issues))

    def test_invalid_fps_is_blocking(self):
        plan = {Path("a.mp4"): _settings(scale=0.1)}
        issues, _ = self._run(plan, prober=_prober(fps=0.0))
        self.assertTrue(any("Invalid FPS" in i for i in issues))

    def test_variable_frame_rate_is_a_warning(self):
        plan = {Path("a.mp4"): _settings(scale=0.1)}
        issues, warnings = self._run(plan, prober=_prober(measured=14.0))
        self.assertFalse(issues)
        self.assertTrue(any("variable frame rate" in w for w in warnings))
        _, warnings = self._run(plan, prober=_prober(measured=10.02))
        self.assertFalse(any("variable frame rate" in w for w in warnings))

    def test_existing_output_files_do_not_warn(self):
        for index in range(15):
            (self.out / f"old_{index}.txt").write_text("x")
        plan = {Path("a.mp4"): _settings(scale=0.1)}
        issues, warnings = self._run(plan, prober=_prober())
        self.assertFalse(issues)
        self.assertFalse(any("overwritten" in w for w in warnings))

    def test_roi_geometry_mismatch_names_the_video(self):
        plan = {Path("a.mp4"): _settings(scale=0.1)}
        issues, _ = self._run(plan, prober=_prober(width=64, height=64))
        self.assertTrue(any(i.startswith("a.mp4: ROI geometry") for i in issues))

    def test_time_range_beyond_video_is_blocking(self):
        plan = {Path("a.mp4"): _settings(scale=0.1, start_time=0.0, end_time=99.0)}
        issues, _ = self._run(plan, prober=_prober())  # 10 s video
        self.assertTrue(any("exceeds" in i for i in issues))

    def test_mixed_calibration_is_flagged(self):
        plan = {Path("a.mp4"): _settings(scale=0.1),
                Path("b.mp4"): _settings(scale=0.0)}
        _, warnings = self._run(plan, prober=_prober())
        self.assertTrue(any("calibrated (cm) and others" in w for w in warnings))

    def test_empty_plan_is_blocking(self):
        issues, _ = self._run({}, prober=_prober())
        self.assertIn("No videos selected for analysis", issues)

    def test_progress_callback_reports_every_video(self):
        calls = []
        plan = {Path(f"v{i}.mp4"): _settings(scale=0.1) for i in range(2)}
        self._run(plan, prober=_prober(),
                  progress=lambda done, total, name: calls.append((done, total)))
        self.assertEqual(calls[0], (0, 2))
        self.assertEqual(calls[-1], (2, 2))


class ProbeVideoTests(unittest.TestCase):
    def test_real_video_is_probed_in_one_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32)
            )
            for _ in range(90):
                writer.write(np.full((32, 32, 3), 100, dtype=np.uint8))
            writer.release()
            probe = probe_video(path)
        self.assertTrue(probe.opened)
        self.assertEqual((probe.width, probe.height), (32, 32))
        self.assertAlmostEqual(probe.fps, 30.0)
        self.assertEqual(probe.frame_count, 90)
        self.assertAlmostEqual(probe.measured_duration_s, 3.0, places=1)
        self.assertIsNone(
            vfr_warning(probe.duration_s, probe.measured_duration_s, probe.fps)
        )

    def test_missing_video_is_reported_not_raised(self):
        probe = probe_video(Path("does_not_exist.mp4"))
        self.assertFalse(probe.exists)
        issues, _ = PreflightChecker().validate_batch(
            {Path("does_not_exist.mp4"): _settings(scale=0.1)}, None,
        )
        self.assertIn("does_not_exist.mp4: Video not found", issues)


class VfrHelperTests(unittest.TestCase):
    def test_matching_timestamps_are_not_flagged(self):
        self.assertIsNone(vfr_warning(100.0, 100.4, 30.0))

    def test_missing_measurement_is_ignored(self):
        self.assertIsNone(vfr_warning(100.0, None, 30.0))
        self.assertIsNone(vfr_warning(None, 100.0, 30.0))

    def test_large_disagreement_is_flagged(self):
        self.assertIn("variable frame rate", vfr_warning(100.0, 110.0, 30.0))


if __name__ == "__main__":
    unittest.main()
