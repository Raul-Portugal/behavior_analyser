"""The validation video draws the open-field center outline from every entry
point, not only from the GUI.

``MainWindow`` refreshes ``ROIManager.center_overlay`` before a run; the CLI
and the child processes of a parallel batch never go through it, so the
pipeline refreshes it itself before drawing the validation video.
"""
import contextlib
import io
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np

    from core import cli
    from core.models import BatchSettings
    from mazes.freestyle import Freestyle
    from tests import test_arena_end_to_end as arena_fixture
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

VIDEO_NAMES = ("openfield_a.avi", "openfield_b.avi")


def _arena_maze() -> "Freestyle":
    maze = Freestyle()
    maze.configure_arena(True, "distance_cm", arena_fixture.CENTER_DISTANCE_CM)
    return maze


@unittest.skipUnless(HAS_RUNTIME, "Scientific runtime is not installed")
class ArenaOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name).resolve()
        first = cls.root / VIDEO_NAMES[0]
        arena_fixture._write_video(first)
        shutil.copy(first, cls.root / VIDEO_NAMES[1])
        cls.maze = _arena_maze()
        settings = arena_fixture._settings(with_arena=True)
        settings.create_timelapse = True
        settings.paradigm_id = cls.maze.name
        settings.algorithm_parameters = cls.maze.get_algorithm_parameters()
        cls.plan_file = cls.root / "plan.json"
        BatchSettings.save_batch_plan(
            {name: settings.copy() for name in VIDEO_NAMES}, cls.plan_file
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _cli(self, label, *extra):
        out = self.root / label
        cache = str(self.root / f"cache_{label}")
        with patch.dict(os.environ, {"LOCALAPPDATA": cache,
                                     "XDG_CACHE_HOME": cache}), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([
                "run", "--plan", str(self.plan_file), "--out", str(out),
                "--quiet", *extra,
            ])
        self.assertEqual(code, cli.EXIT_OK)
        (run_dir,) = sorted(out.glob("analysis_run_*"))
        return run_dir

    @staticmethod
    def _timelapse(run_dir: Path, name: str = "openfield_a") -> Path:
        return run_dir / name / "media" / f"{name}_timelapse.mp4"

    @staticmethod
    def _first_frame(path: Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        assert ok, f"cannot read {path}"
        return frame

    def test_cli_run_draws_the_center_outline(self):
        calls = []
        real = Freestyle.update_roi_overlay

        def spy(maze, roi_manager, scale_factor):
            calls.append(scale_factor)
            return real(maze, roi_manager, scale_factor)

        with patch.object(Freestyle, "update_roi_overlay", spy):
            drawn = self._cli("with_overlay", "--workers", "1")
        self.assertTrue(calls)
        self.assertAlmostEqual(calls[0], arena_fixture.SCALE_CM_PER_PX)

        with patch.object(Freestyle, "update_roi_overlay",
                          lambda maze, roi_manager, scale_factor: None):
            plain = self._cli("without_overlay", "--workers", "1")

        with_overlay = self._first_frame(self._timelapse(drawn)).astype(int)
        without = self._first_frame(self._timelapse(plain)).astype(int)
        changed = np.abs(with_overlay - without).max(axis=2) > 40
        # The center outline is a closed polygon well over a hundred pixels
        # long; the arena outline and the animal are identical in both runs.
        self.assertGreater(int(changed.sum()), 100)

    def test_parallel_child_processes_draw_the_same_overlay(self):
        sequential = self._cli("sequential", "--workers", "1")
        parallel = self._cli("parallel", "--workers", "2")
        for name in ("openfield_a", "openfield_b"):
            self.assertEqual(
                self._timelapse(parallel, name).read_bytes(),
                self._timelapse(sequential, name).read_bytes(),
                name,
            )

        with patch.object(Freestyle, "update_roi_overlay",
                          lambda maze, roi_manager, scale_factor: None):
            plain = self._cli("plain_sequential", "--workers", "1")
        drawn = self._first_frame(self._timelapse(parallel)).astype(int)
        undrawn = self._first_frame(self._timelapse(plain)).astype(int)
        self.assertGreater(
            int((np.abs(drawn - undrawn).max(axis=2) > 40).sum()), 100
        )

    def test_center_overlay_survives_pickling(self):
        settings = arena_fixture._settings(with_arena=True)
        self.maze.update_roi_overlay(
            settings.roi_manager, arena_fixture.SCALE_CM_PER_PX
        )
        self.assertIsNotNone(settings.roi_manager.center_overlay)
        copy = pickle.loads(pickle.dumps(settings))
        self.assertTrue(np.array_equal(
            copy.roi_manager.center_overlay, settings.roi_manager.center_overlay
        ))

    def test_a_run_without_an_arena_draws_no_center_outline(self):
        settings = arena_fixture._settings(with_arena=False)
        maze = Freestyle()
        maze.update_roi_overlay(settings.roi_manager, 0.05)
        self.assertIsNone(settings.roi_manager.center_overlay)


if __name__ == "__main__":
    unittest.main()
