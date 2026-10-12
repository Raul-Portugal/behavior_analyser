import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False

from core.batch_plan_keys import build_plan_by_key
from core.models import AppConfig, BatchSettings
from roi_manager import ROIManager


def _settings(tag: int) -> BatchSettings:
    manager = ROIManager()
    manager.add_roi("arm_a", np.array([[0, 0], [50, 0], [50, 50 + tag]]))
    return BatchSettings(manager, AppConfig().detection, 0.0)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class BatchPlanWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        self.plan_file = root / "plan.json"
        self.a = root / "day1" / "mouse1.mp4"
        self.b = root / "day2" / "mouse1.mp4"
        self.new = root / "day3" / "mouse9.mp4"

        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        from mazes import create_maze
        self.window.maze = create_maze("Y-Maze")
        self.window.batch_plan_cache = build_plan_by_key(
            {self.a: _settings(0), self.b: _settings(40)}, self.plan_file
        )
        self.window.batch_plan_path = self.plan_file
        self.window.video_paths = [self.b, self.new, self.a]

    def _run(self, choice):
        match = self.window._match_plan(self.window.video_paths)
        with patch.object(self.window, "_ask_unmatched_videos",
                          return_value=choice), \
                patch.object(self.window, "run_worker") as run_worker:
            self.window._start_from_batch_plan(match, "Y-Maze")
        return match, run_worker

    def test_same_named_videos_get_their_own_settings(self):
        match, _ = self._run("cancel")
        self.assertEqual(set(match.matched), {self.a, self.b})
        self.assertEqual([v for v, _ in match.unmatched], [self.new])
        key_a, key_b = match.matched[self.a], match.matched[self.b]
        self.assertNotEqual(key_a, key_b)

    def test_skip_runs_only_matched_videos_in_list_order(self):
        _, run_worker = self._run("skip")
        run_worker.assert_called_once()
        plan = run_worker.call_args.args[0]
        self.assertEqual(list(plan), [self.b, self.a])
        self.assertNotIn(self.new, plan)
        self.assertNotEqual(
            plan[self.a].roi_manager.rois["arm_a"][0].points.tolist(),
            plan[self.b].roi_manager.rois["arm_a"][0].points.tolist(),
        )

    def test_cancel_runs_nothing(self):
        _, run_worker = self._run("cancel")
        run_worker.assert_not_called()

    def test_unmatched_videos_never_inherit_another_videos_settings(self):
        self.window.video_paths = [self.new]
        match = self.window._match_plan(self.window.video_paths)
        self.assertEqual(match.matched, {})
        self.assertIsNone(self.window._plan_settings_for(self.new))


if __name__ == "__main__":
    unittest.main()
