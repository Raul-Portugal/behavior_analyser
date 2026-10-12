"""Append-only checkpoints: format, I/O growth and resume equivalence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from checkpoint_manager import CheckpointManager

try:
    import cv2
    import numpy as np

    from core.models import BatchSettings, DetectionConfig
    from core.video import ReferenceFrameGenerator
    from gui.worker import AnalysisWorker
    from mazes.freestyle import Freestyle
    from mazes.tst import TST
    from roi_manager import ROIManager
    from tests import tst_synthetic as synthetic
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


class FakeSettings:
    def to_dict(self):
        return {"threshold": 1.0}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _tst_like_result(rows: int) -> dict:
    """A partial result shaped like TSTResult mid-run."""
    return {
        "timestamps": [i / 30.0 for i in range(rows)],
        "activity_scores": {"mouse_1": [0.5 + i for i in range(rows)]},
        "motion_evidence": {
            "mouse_1": [
                {
                    "state": "mobile",
                    "confidence": 0.9,
                    "mask_contour": [[i, i]] * 40,
                }
                for i in range(rows)
            ]
        },
        "positions": [(float(i), 1.0) if i % 7 else None for i in range(rows)],
        "visual_labels": [],
        "time_in_roi": {},
        "fps": 30.0,
        "total_distance": 12.5,
        "_tst_engine_state": {"histories": {"mouse_1": {"centroid": [1, 2]}}},
    }


class CheckpointFormatTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.video = self.root / "trial.mp4"
        self.video.write_bytes(b"video-data")
        self.settings = FakeSettings()

    def _manager(self, **kwargs):
        return CheckpointManager(self.root, **kwargs)

    def test_per_frame_data_round_trips_and_contours_are_omitted(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 49, _tst_like_result(50), self.settings
        )

        restored = self._manager().load_checkpoint(self.video, self.settings)

        self.assertEqual(restored.last_frame_processed, 49)
        partial = restored.partial_result
        self.assertEqual(len(partial["timestamps"]), 50)
        self.assertEqual(
            partial["activity_scores"]["mouse_1"][:3], [0.5, 1.5, 2.5]
        )
        self.assertIsNone(partial["positions"][0])
        self.assertEqual(partial["positions"][1], [1.0, 1.0])
        evidence = partial["motion_evidence"]["mouse_1"]
        self.assertEqual(len(evidence), 50)
        self.assertEqual(evidence[3], {"state": "mobile", "confidence": 0.9})
        # Scalars and temporal state travel in the header.
        self.assertEqual(partial["visual_labels"], [])
        self.assertEqual(partial["total_distance"], 12.5)
        self.assertEqual(
            partial["_tst_engine_state"]["histories"]["mouse_1"]["centroid"],
            [1, 2],
        )

    def test_header_stays_small_and_rows_go_to_the_sidecar(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 999, _tst_like_result(1000), self.settings
        )
        header_path = manager._get_checkpoint_path(self.video)
        rows_path = manager._get_rows_path(header_path)

        header = json.loads(header_path.read_text(encoding="utf-8"))
        self.assertEqual(header["version"], CheckpointManager.CHECKPOINT_VERSION)
        self.assertEqual(header["rows_committed"], 1000)
        self.assertLess(header_path.stat().st_size, 2000)
        self.assertEqual(
            len(rows_path.read_bytes().splitlines()), 1000
        )

    def test_bytes_written_grow_linearly_with_total_rows(self):
        total_rows, saves = 2000, 20
        incremental = self._manager()
        full = result = _tst_like_result(total_rows)
        step = total_rows // saves
        for save in range(1, saves + 1):
            rows = save * step
            partial = {
                key: (
                    value[:rows] if isinstance(value, list) and len(value) == total_rows
                    else {k: v[:rows] for k, v in value.items()}
                    if key in ("activity_scores", "motion_evidence")
                    else value
                )
                for key, value in result.items()
            }
            incremental.save_checkpoint(
                self.video, rows - 1, partial, self.settings
            )

        single_dir = self.root / "single"
        single_dir.mkdir()
        single = CheckpointManager(single_dir)
        single.save_checkpoint(self.video, total_rows - 1, full, self.settings)

        # N checkpoints cost about one full write plus N small headers, not
        # N x rows (which would be roughly saves / 2 times larger).
        self.assertLess(incremental.bytes_written, single.bytes_written * 1.5)
        rows_path = incremental._get_rows_path(
            incremental._get_checkpoint_path(self.video)
        )
        self.assertEqual(len(rows_path.read_bytes().splitlines()), total_rows)

    def test_torn_trailing_line_is_ignored(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 29, _tst_like_result(30), self.settings
        )
        rows_path = manager._get_rows_path(manager._get_checkpoint_path(self.video))
        with open(rows_path, "ab") as f:
            f.write(b'{"timestamps":0.5,"positions":[1.0,')  # crash mid-line

        restored = self._manager().load_checkpoint(self.video, self.settings)

        self.assertIsNotNone(restored)
        self.assertEqual(len(restored.partial_result["timestamps"]), 30)

    def test_uncommitted_complete_rows_are_ignored_and_overwritten(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 29, _tst_like_result(30), self.settings
        )
        rows_path = manager._get_rows_path(manager._get_checkpoint_path(self.video))
        # Rows fsynced but the header never rewritten (crash in between).
        with open(rows_path, "ab") as f:
            f.write(b'{"junk":1}\n' * 5)

        resumer = self._manager()
        restored = resumer.load_checkpoint(self.video, self.settings)
        self.assertEqual(len(restored.partial_result["timestamps"]), 30)

        resumer.save_checkpoint(
            self.video, 39, _tst_like_result(40), self.settings
        )
        again = self._manager().load_checkpoint(self.video, self.settings)
        self.assertEqual(len(again.partial_result["timestamps"]), 40)
        self.assertNotIn(b"junk", rows_path.read_bytes())

    def test_missing_committed_rows_invalidate_the_checkpoint(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 29, _tst_like_result(30), self.settings
        )
        rows_path = manager._get_rows_path(manager._get_checkpoint_path(self.video))
        lines = rows_path.read_bytes().splitlines(keepends=True)
        rows_path.write_bytes(b"".join(lines[:20]))

        self.assertIsNone(
            self._manager().load_checkpoint(self.video, self.settings)
        )

    def test_changed_row_layout_rewrites_the_sidecar(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 9, _tst_like_result(10), self.settings
        )
        result = _tst_like_result(20)
        result["extra_trace"] = list(range(20))
        manager.save_checkpoint(self.video, 19, result, self.settings)

        restored = self._manager().load_checkpoint(self.video, self.settings)
        self.assertEqual(restored.partial_result["extra_trace"], list(range(20)))
        self.assertEqual(len(restored.partial_result["timestamps"]), 20)

    def test_clear_removes_header_and_sidecar(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 9, _tst_like_result(10), self.settings
        )
        header = manager._get_checkpoint_path(self.video)
        manager.clear_checkpoint(self.video)
        self.assertFalse(header.exists())
        self.assertFalse(manager._get_rows_path(header).exists())

    def test_old_checkpoint_version_is_ignored(self):
        manager = self._manager()
        manager.save_checkpoint(
            self.video, 9, _tst_like_result(10), self.settings
        )
        header_path = manager._get_checkpoint_path(self.video)
        header = json.loads(header_path.read_text(encoding="utf-8"))
        header["version"] = 3
        header_path.write_text(json.dumps(header), encoding="utf-8")
        self.assertIsNone(
            self._manager().load_checkpoint(self.video, self.settings)
        )

    def test_save_interval_is_wall_clock_based_and_configurable(self):
        clock = FakeClock()
        manager = self._manager(interval_seconds=30.0, clock=clock)
        self.assertFalse(manager.should_save_checkpoint())
        clock.now = 29.0
        self.assertFalse(manager.should_save_checkpoint())
        clock.now = 31.0
        self.assertTrue(manager.should_save_checkpoint())
        manager.save_checkpoint(
            self.video, 9, _tst_like_result(10), self.settings
        )
        self.assertFalse(manager.should_save_checkpoint())
        clock.now = 62.0
        self.assertTrue(manager.should_save_checkpoint())
        manager.reset_timer()
        self.assertFalse(manager.should_save_checkpoint())


def _cancel_at_first_progress(worker):
    state = {"fired": False}

    def on_progress(_pct, _main, sub):
        if not state["fired"] and ("Frame " in sub or "Motion Analysis" in sub):
            state["fired"] = True
            worker.stop()

    worker.detailed_progress.connect(on_progress)


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class WorkerResumeEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        cache = patch.object(
            ReferenceFrameGenerator, "get_cache_dir",
            return_value=self.root / "cache",
        )
        cache.start()
        self.addCleanup(cache.stop)

    def _run(self, maze_factory, settings_factory, video, out_dir, cancel=False):
        out_dir.mkdir(exist_ok=True)
        worker = AnalysisWorker(
            {video: settings_factory()}, out_dir, maze_factory()
        )
        logs = []
        worker.log.connect(logs.append)
        outcomes = []
        worker.finished.connect(outcomes.append)
        if cancel:
            _cancel_at_first_progress(worker)
        worker.run()
        self.assertEqual(len(outcomes), 1)
        return worker, outcomes[0], logs

    # -- tracked path -------------------------------------------------
    def _tracking_video(self):
        path = self.root / "blob.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120)
        )
        for index in range(100):
            frame = np.full((120, 160, 3), 40, dtype=np.uint8)
            cx = 30 + index
            cy = 60 + int(10 * np.sin(index / 9.0))
            cv2.rectangle(
                frame, (cx - 12, cy - 12), (cx + 12, cy + 12),
                (255, 255, 255), -1,
            )
            writer.write(frame)
        writer.release()
        return path

    @staticmethod
    def _tracking_settings():
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi(
            "arena", np.array([[0, 0], [159, 0], [159, 119], [0, 119]])
        )
        return BatchSettings(
            manager,
            DetectionConfig(
                threshold_percentile=95.0, use_weighting=False,
                min_component_area=20, morphology_kernel_size=3,
                min_detection_confidence=0.1,
            ),
            0.0,
        )

    def test_tracked_resume_matches_uninterrupted_run(self):
        video = self._tracking_video()
        _, full, _ = self._run(
            Freestyle, self._tracking_settings, video, self.root / "full"
        )
        self.assertEqual(full.failures, [])

        resume_dir = self.root / "resume"
        first, cancelled, _ = self._run(
            Freestyle, self._tracking_settings, video, resume_dir, cancel=True
        )
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(
            len(first.checkpoint_mgr.find_all_checkpoints()), 1
        )

        _, resumed, logs = self._run(
            Freestyle, self._tracking_settings, video, resume_dir
        )
        self.assertTrue(any("Resuming from checkpoint" in m for m in logs), logs)
        self.assertEqual(resumed.failures, [])

        expected = full.results[0][1]
        actual = resumed.results[0][1]
        self.assertEqual(len(actual.positions), len(expected.positions))
        self.assertGreater(len(actual.positions), 90)
        self.assertEqual(actual.timestamps, expected.timestamps)
        self.assertEqual(
            [tuple(p) if p else None for p in actual.positions],
            [tuple(p) if p else None for p in expected.positions],
        )
        self.assertEqual(actual.roi_labels, expected.roi_labels)
        self.assertEqual(actual.detection_confidences, expected.detection_confidences)
        self.assertAlmostEqual(actual.total_distance, expected.total_distance)
        self.assertEqual(
            len(first.checkpoint_mgr.find_all_checkpoints()), 0,
            "checkpoint must be cleared after a completed run",
        )

    # -- TST path -----------------------------------------------------
    def _tst_video(self):
        path = self.root / "tst.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
            synthetic.BASE_SIZE,
        )
        for frame in synthetic.render_clip(30.0, duration=4.0):
            writer.write(frame)
        writer.release()
        return path

    @staticmethod
    def _tst_settings():
        manager = ROIManager(source_dimensions=synthetic.BASE_SIZE)
        manager.add_roi("mouse_1", synthetic.clip_roi_points())
        return BatchSettings(
            manager, DetectionConfig(), 0.0,
            start_time=0.0, end_time=4.0,
            per_roi_times={"mouse_1": (0.0, 4.0)},
            paradigm_id="Tail Suspension Test",
            algorithm_parameters=TST().get_algorithm_parameters(),
        )

    def test_tst_resume_matches_uninterrupted_run(self):
        video = self._tst_video()
        _, full, _ = self._run(
            TST, self._tst_settings, video, self.root / "tst_full"
        )
        self.assertEqual(full.failures, [])

        resume_dir = self.root / "tst_resume"
        first, cancelled, _ = self._run(
            TST, self._tst_settings, video, resume_dir, cancel=True
        )
        self.assertTrue(cancelled.cancelled)
        checkpoints = first.checkpoint_mgr.find_all_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        header = json.loads(checkpoints[0].read_text(encoding="utf-8"))
        # The header carries the engine state and no per-frame arrays.
        self.assertIn("_tst_engine_state", header["partial_result"])
        self.assertLess(checkpoints[0].stat().st_size, 20000)
        self.assertNotIn(
            "mask_contour",
            checkpoints[0].with_suffix(".rows.jsonl").read_text(encoding="utf-8"),
        )

        _, resumed, logs = self._run(
            TST, self._tst_settings, video, resume_dir
        )
        self.assertTrue(any("Resuming from checkpoint" in m for m in logs), logs)
        self.assertEqual(resumed.failures, [])

        expected = full.results[0][1]
        actual = resumed.results[0][1]
        self.assertEqual(actual.timestamps, expected.timestamps)
        self.assertGreater(len(actual.timestamps), 100)
        self.assertEqual(actual.behavior_states, expected.behavior_states)
        self.assertEqual(actual.immobile_states, expected.immobile_states)
        np.testing.assert_allclose(
            actual.activity_scores["mouse_1"],
            expected.activity_scores["mouse_1"],
            rtol=1e-9, atol=1e-9,
        )
        self.assertEqual(
            actual.total_immobility_time, expected.total_immobility_time
        )

        def strip(trace):
            return [
                {k: v for k, v in item.items() if k != "mask_contour"}
                for item in trace
            ]

        self.assertEqual(
            strip(actual.motion_evidence["mouse_1"]),
            strip(expected.motion_evidence["mouse_1"]),
        )


if __name__ == "__main__":
    unittest.main()
