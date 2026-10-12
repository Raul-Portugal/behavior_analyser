"""What the TST production loop records per frame, and how fast it gets there."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    import cv2

    from core.tst_motion_engine import TSTMotionEngine
    from gui.worker import AnalysisWorker
    from mazes.tst import TST
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


if __name__ == "__main__":
    unittest.main()
