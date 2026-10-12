"""What the TST production loop records per frame, and how fast it gets there."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    import cv2

    from core.tst_motion_engine import TST_VALIDATION_FRAME_STEP, TSTMotionEngine
    from gui.worker import AnalysisWorker
    from mazes.tst import TST
    from output import DataExporter
    from tests import worker_support as support
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def decoded_frames(video_path, first_frame):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames


class WorkerRun:
    """Drive ``_run_tst_analysis`` directly on a synthetic TST video."""

    def __init__(self, root, **settings_options):
        self.root = Path(root)
        self.video = support.write_tst_video(self.root)
        self.settings = support.tst_settings(**settings_options)
        output_dir = self.root / "out"
        output_dir.mkdir()
        self.worker = AnalysisWorker({}, output_dir, TST())

    def run(self, start_frame=0, end_frame=120, **kwargs):
        return self.worker._run_tst_analysis(
            self.video, self.settings, start_frame, end_frame,
            Mock(fps=30.0), Mock(), 0, lambda pct: pct, "clip", **kwargs
        )

    def serial_observations(self, start_frame, end_frame):
        roi_dict = {
            name: rois[0]
            for name, rois in self.settings.roi_manager.rois.items() if rois
        }
        engine = TSTMotionEngine(
            roi_dict, self.worker.maze.motion_config, fps=30.0
        )
        frames = decoded_frames(self.video, start_frame)
        return [
            engine.calculate_observations(frames[index], frames[index - 1])
            for index in range(1, min(len(frames), end_frame - start_frame))
        ]


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TSTProductionLoopEquivalenceTests(unittest.TestCase):
    def test_pipelined_production_loop_matches_serial_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory)
            result, processed = run.run(start_frame=0, end_frame=120)
            expected = run.serial_observations(0, 120)

            self.assertEqual(processed, len(expected))
            self.assertGreater(processed, 100)
            fields = [
                {key: value for key, value in item.to_dict().items()}
                for pair in expected for item in pair.values()
            ]
            recorded = [
                {key: item[key] for key in fields[0]}
                for item in result.motion_evidence["mouse_1"]
            ]
            self.assertEqual(recorded, fields)
            self.assertEqual(
                result.activity_scores["mouse_1"],
                [pair["mouse_1"].activity_score for pair in expected],
            )
            self.assertEqual(
                result.timestamps, [i / 30.0 for i in range(1, 120)]
            )

    def test_start_frame_offset_run_matches_serial_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory)
            result, processed = run.run(start_frame=40, end_frame=110)
            expected = run.serial_observations(40, 110)
            self.assertEqual(processed, len(expected))
            self.assertEqual(
                [item["state"] for item in result.motion_evidence["mouse_1"]],
                [pair["mouse_1"].state for pair in expected],
            )
            self.assertAlmostEqual(result.timestamps[0], 41 / 30.0)

    def test_cancelled_run_checkpoints_only_consumed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory)
            manager = run.worker.checkpoint_mgr
            saved = []
            manager.save_checkpoint = Mock(
                side_effect=lambda video, frame_idx, payload, settings: (
                    saved.append((frame_idx, dict(payload)))
                )
            )
            checks = []

            def should_save():
                # Called once per consumed frame: cancel after the 12th.
                checks.append(1)
                if len(checks) == 12:
                    run.worker.is_cancelled = True
                return False

            manager.should_save_checkpoint = should_save
            with self.assertRaises(InterruptedError):
                run.run(start_frame=0, end_frame=120)

            self.assertEqual(len(saved), 1)
            frame_idx, payload = saved[0]
            self.assertEqual(frame_idx, 12)
            self.assertEqual(len(payload["timestamps"]), 12)
            # Up to 2 x workers later frames were already in flight; the
            # saved engine state must still be exactly that of frame 12.
            reference = TSTMotionEngine(
                {"mouse_1": run.settings.roi_manager.rois["mouse_1"][0]},
                run.worker.maze.motion_config, fps=30.0,
            )
            frames = decoded_frames(run.video, 0)
            for index in range(1, 13):
                reference.calculate_observations(
                    frames[index], frames[index - 1]
                )
            self.assertEqual(
                payload["_tst_engine_state"], reference.export_state()
            )


class _SpyEvidence(dict):
    """Evidence row that records whether a contour was looked up."""

    def __init__(self, index, row, log):
        super().__init__(row)
        self._index, self._log = index, log

    def get(self, key, default=None):
        value = super().get(key, default)
        if key == "mask_contour":
            self._log.append((self._index, value))
        return value


@unittest.skipUnless(HAS_RUNTIME, "GUI/scientific runtime is not installed")
class TSTContourStorageTests(unittest.TestCase):
    END = 100

    def test_contours_are_not_stored_without_a_validation_video(self):
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory, create_timelapse=False)
            result, _ = run.run(end_frame=self.END)
        self.assertGreater(len(result.motion_evidence["mouse_1"]), 90)
        for item in result.motion_evidence["mouse_1"]:
            self.assertNotIn("mask_contour", item)

    def test_only_sampled_frames_keep_a_contour_when_requested(self):
        self.assertEqual(TST_VALIDATION_FRAME_STEP, 3)
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory, create_timelapse=True)
            result, _ = run.run(end_frame=self.END)
            roi = run.settings.roi_manager.rois["mouse_1"][0]
            engine = TSTMotionEngine(
                {"mouse_1": roi}, run.worker.maze.motion_config, fps=30.0
            )
            frames = decoded_frames(run.video, 0)
            expected = {}
            for index in range(1, self.END):
                engine.calculate_observations(frames[index], frames[index - 1])
                expected[index - 1] = engine.get_last_mask_contour("mouse_1")

        evidence = result.motion_evidence["mouse_1"]
        for index, item in enumerate(evidence):
            if index % TST_VALIDATION_FRAME_STEP == 0:
                self.assertEqual(item["mask_contour"], expected[index], index)
            else:
                self.assertNotIn("mask_contour", item, index)
        sampled = [
            item for index, item in enumerate(evidence)
            if index % TST_VALIDATION_FRAME_STEP == 0
        ]
        self.assertGreater(
            sum(len(item["mask_contour"]) >= 3 for item in sampled),
            len(sampled) - 3,
        )

    def test_contour_sampling_does_not_change_any_other_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with_video, _ = WorkerRun(
                directory, create_timelapse=True
            ).run(end_frame=self.END)
        with tempfile.TemporaryDirectory() as directory:
            without_video, _ = WorkerRun(
                directory, create_timelapse=False
            ).run(end_frame=self.END)

        def strip(result):
            return [
                {k: v for k, v in item.items() if k != "mask_contour"}
                for item in result.motion_evidence["mouse_1"]
            ]

        self.assertEqual(strip(with_video), strip(without_video))

    def test_exporters_accept_evidence_without_contours(self):
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory, create_timelapse=False)
            result, _ = run.run(end_frame=self.END)
            run.worker.maze.calculate_metrics(result)
            csv_path = Path(directory) / "evidence.csv"
            DataExporter.export_to_csv(result, csv_path)
            lines = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines) - 1, len(result.timestamps))

    def test_validation_video_frames_still_get_their_contours(self):
        with tempfile.TemporaryDirectory() as directory:
            run = WorkerRun(directory, create_timelapse=True)
            result, _ = run.run(end_frame=self.END)
            run.worker.maze.calculate_metrics(result)
            lookups = []
            result.motion_evidence["mouse_1"] = [
                _SpyEvidence(index, item, lookups)
                for index, item in enumerate(result.motion_evidence["mouse_1"])
            ]
            media_dir = Path(directory) / "media"
            media_dir.mkdir()
            run.worker._generate_tst_validation_video(
                result, run.video, run.settings, "clip", media_dir
            )
            self.assertTrue((media_dir / "clip_validation.mp4").exists())
        self.assertGreater(len(lookups), 20)
        for index, contour in lookups:
            self.assertEqual(index % TST_VALIDATION_FRAME_STEP, 0, index)
            self.assertGreaterEqual(len(contour), 3, index)


if __name__ == "__main__":
    unittest.main()
