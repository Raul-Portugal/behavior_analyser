import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core.tst_motion_engine import TSTMotionConfig, TSTMotionEngine
    from roi_manager import ROI
    from tests import tst_synthetic as synthetic
    HAS_OPENCV = True
except ModuleNotFoundError:
    HAS_OPENCV = False

try:
    from PyQt6.QtWidgets import QApplication

    from gui.tst_tuner import EnergyScanner
    HAS_GUI = HAS_OPENCV
except ModuleNotFoundError:
    HAS_GUI = False

FULL_SIZE = (1920, 1080)


def _full_resolution(frame):
    return cv2.resize(frame, FULL_SIZE, interpolation=cv2.INTER_LINEAR)


def _scaled_roi(name, scale_x, scale_y):
    points = synthetic.clip_roi_points().astype(np.float64)
    points *= [scale_x, scale_y]
    return ROI(np.rint(points).astype(np.int32), name)


@unittest.skipUnless(HAS_OPENCV, "OpenCV runtime is not installed")
class TSTCanonicalGeometryTests(unittest.TestCase):
    def setUp(self):
        self.times = [2.3 + index / 30.0 for index in range(14)]
        base = [synthetic.render_frame(t) for t in self.times]
        self.full = [_full_resolution(frame) for frame in base]
        self.small = [
            cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            for frame in self.full
        ]
        self.full_roi = _scaled_roi(
            "mouse_1", FULL_SIZE[0] / 400.0, FULL_SIZE[1] / 300.0
        )
        self.small_roi = ROI(
            np.rint(
                self.full_roi.points.astype(np.float64) * 0.25
            ).astype(np.int32),
            "mouse_1",
        )

    def _run(self, roi, frames):
        engine = TSTMotionEngine({"mouse_1": roi}, fps=30.0)
        observations = []
        contours = []
        for previous, current in zip(frames, frames[1:]):
            observations.append(
                engine.calculate_observations(current, previous)["mouse_1"]
            )
            contours.append(engine.get_last_mask_contour("mouse_1"))
        return observations, contours

    def test_full_resolution_matches_preresized_frames(self):
        full_obs, full_contours = self._run(self.full_roi, self.full)
        small_obs, small_contours = self._run(self.small_roi, self.small)

        for left, right in zip(full_obs, small_obs):
            self.assertEqual(left.state, right.state)
            for name in (
                "rigid_motion", "rotation_motion", "residual_motion",
                "deformation", "coherence", "motion_injection",
                "pendulum_score", "mask_area_ratio",
            ):
                self.assertAlmostEqual(
                    getattr(left, name), getattr(right, name), places=6,
                    msg=name,
                )
        # Contours are reported in source pixels: 4x the 480-wide ones.
        for left, right in zip(full_contours, small_contours):
            self.assertEqual(
                left, [[point[0] * 4, point[1] * 4] for point in right]
            )

    def test_contour_is_mapped_to_source_pixels(self):
        _, contours = self._run(self.full_roi, self.full)
        points = np.asarray(contours[-1])
        self.assertGreaterEqual(len(points), 3)
        roi_points = self.full_roi.points
        self.assertTrue(np.all(points[:, 0] >= roi_points[:, 0].min()))
        self.assertTrue(np.all(points[:, 0] <= roi_points[:, 0].max()))
        self.assertTrue(np.all(points[:, 1] >= roi_points[:, 1].min()))
        self.assertTrue(np.all(points[:, 1] <= roi_points[:, 1].max()))
        # The mouse is in the middle of the ROI, not in a crop-origin corner.
        centre = points.mean(axis=0)
        self.assertAlmostEqual(centre[0], 200.0 * 4.8, delta=120)
        self.assertAlmostEqual(centre[1], 150.0 * 3.6, delta=120)

    def test_narrow_frames_are_not_upscaled(self):
        roi = ROI(synthetic.clip_roi_points(), "mouse_1")
        engine = TSTMotionEngine({"mouse_1": roi}, analysis_width=480)
        frame = synthetic.render_frame(2.0)
        engine.calculate_observations(frame, frame)
        self.assertEqual(engine._analysis_size, (400, 300))
        self.assertEqual(engine._scale, (1.0, 1.0))

    def test_processing_is_confined_to_the_expanded_roi_box(self):
        engine = TSTMotionEngine({"mouse_1": self.full_roi}, fps=30.0)
        shapes = []
        original = cv2.calcOpticalFlowFarneback

        def recorded(previous, current, *args, **kwargs):
            shapes.append((previous.shape, current.shape))
            return original(previous, current, *args, **kwargs)

        with patch(
            "core.tst_motion_engine.cv2.calcOpticalFlowFarneback",
            side_effect=recorded,
        ):
            engine.calculate_observations(self.full[1], self.full[0])
        x0, y0, x1, y1 = engine._crop
        self.assertEqual(shapes, [((y1 - y0, x1 - x0),) * 2])
        # 480x270 analysis frame; the ROI box plus 15% margin is smaller.
        self.assertEqual(engine._analysis_size, (480, 270))
        self.assertLess((x1 - x0) * (y1 - y0), 480 * 270)
        # The crop contains the whole ROI.
        scaled = self.full_roi.points * 0.25
        self.assertLessEqual(x0, scaled[:, 0].min())
        self.assertGreaterEqual(x1, scaled[:, 0].max())

    def test_analysis_width_is_part_of_the_validated_config(self):
        self.assertEqual(TSTMotionConfig().analysis_width, 480)
        with self.assertRaises(ValueError):
            TSTMotionConfig(analysis_width=64).validate()
        with self.assertRaises(ValueError):
            TSTMotionConfig(analysis_width=480.5).validate()


@unittest.skipUnless(HAS_GUI, "Qt/OpenCV runtime is not installed")
class TSTScannerProductionAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _write_video(self, directory, fps=30.0):
        path = Path(directory) / "synthetic_tst.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps,
            (FULL_SIZE[0] // 2, FULL_SIZE[1] // 2),
        )
        self.assertTrue(writer.isOpened())
        for frame in synthetic.render_clip(fps):
            writer.write(cv2.resize(
                frame, (FULL_SIZE[0] // 2, FULL_SIZE[1] // 2),
                interpolation=cv2.INTER_LINEAR,
            ))
        writer.release()
        return path

    def _production_states(self, path, roi):
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        engine = TSTMotionEngine({"mouse_1": roi}, fps=fps)
        states = {}
        ret, previous = capture.read()
        index = 0
        while ret:
            ret, frame = capture.read()
            if not ret:
                break
            index += 1
            states[index] = engine.calculate_observations(
                frame, previous
            )["mouse_1"].state
            previous = frame
        capture.release()
        return states

    def _scan(self, path, roi, target_fps):
        scanner = EnergyScanner(
            path, {"mouse_1": roi}, TSTMotionConfig(),
            [(0, 250)], target_fps=target_fps,
        )
        completed = []
        scanner.scan_complete.connect(lambda *args: completed.append(args))
        scanner.run()
        return scanner, completed

    def test_tuner_scan_agrees_with_production_path(self):
        width, height = FULL_SIZE[0] // 2, FULL_SIZE[1] // 2
        roi = _scaled_roi("mouse_1", width / 400.0, height / 300.0)
        thresholds = ((30.0, 0.95), (12.0, 0.90))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_video(directory)
            # The production pass and both scans only read the video, so run
            # them side by side (OpenCV releases the GIL).
            with ThreadPoolExecutor(max_workers=3) as pool:
                production_future = pool.submit(
                    self._production_states, path, roi
                )
                scan_futures = [
                    pool.submit(self._scan, path, roi, target_fps)
                    for target_fps, _ in thresholds
                ]
                production = production_future.result()
                scans = [future.result() for future in scan_futures]
        for (target_fps, minimum), (scanner, completed) in zip(
            thresholds, scans
        ):
            self.assertEqual(len(completed), 1)
            evidence = scanner.evidence_data["mouse_1"]
            self.assertGreater(len(evidence), 20)
            matches = [
                item["state"] == production[item["source_frame"]]
                for item in evidence
                if item["source_frame"] in production
            ]
            self.assertGreaterEqual(
                np.mean(matches), minimum, f"{target_fps} fps scan"
            )


if __name__ == "__main__":
    unittest.main()
