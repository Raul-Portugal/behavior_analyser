"""Cancelling must not silently throw away the user's work."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtCore import QPointF
    from PyQt6.QtWidgets import QApplication
    from core.models import AppConfig, BatchSettings
    from gui import ux_helpers
    from gui.freestyle_config_dialog import FreestyleConfigDialog
    from gui.main_window import MainWindow
    from gui.roi_selector import RoiSelectorDialog
    from gui.settings_dialog import SettingsDialog
    from gui.setup_progress import (KIND_ARMS, KIND_FREESTYLE, KIND_STANDARD,
                                    KIND_TST, STEP_ARM_RULES, STEP_DETECTION,
                                    STEP_MICE, STEP_ROI, STEP_TST_TUNER,
                                    STEP_ZONES, SetupProgress, wizard_steps)
    from gui.tst_tuner import TstTunerDialog
    from gui.tuner_dialog import TunerDialog
    from mazes import create_maze
    from roi_manager import ROIManager
    from tests import worker_support as support
    from tests.gui_support import temp_app_settings
    HAS_RUNTIME = True
except ModuleNotFoundError:
    HAS_RUNTIME = False


def _square(offset=0):
    return np.array([[offset, 0], [offset + 40, 0], [offset + 40, 40], [offset, 40]])


# ----------------------------------------------------------------------
# Pure decisions
# ----------------------------------------------------------------------
@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class DecisionHelperTests(unittest.TestCase):
    def test_should_confirm_discard_only_when_state_changed(self):
        self.assertFalse(ux_helpers.should_confirm_discard((1, "a"), (1, "a")))
        self.assertTrue(ux_helpers.should_confirm_discard((1, "a"), (2, "a")))

    def test_has_drawn_work(self):
        manager = ROIManager()
        self.assertFalse(ux_helpers.has_drawn_work(manager, 0))
        self.assertTrue(ux_helpers.has_drawn_work(manager, 1))
        manager.add_roi("arm_a", _square())
        self.assertTrue(ux_helpers.has_drawn_work(manager, 0))

    def test_no_options_when_nothing_was_accepted_yet(self):
        self.assertEqual(ux_helpers.verification_cancel_choices(0), [])

    def test_partial_verification_choices_name_the_accepted_count(self):
        choices = ux_helpers.verification_cancel_choices(3)
        labels = {choice.key: choice.label for choice in choices}
        self.assertEqual(labels[ux_helpers.VERIFY_RUN], "Run the 3 verified videos")
        self.assertIn("Save partial batch plan", labels[ux_helpers.VERIFY_SAVE])
        self.assertEqual(labels[ux_helpers.VERIFY_DISCARD], "Discard")
        self.assertEqual(
            ux_helpers.verification_cancel_choices(1)[0].label,
            "Run the 1 verified video",
        )

    def test_partial_verification_outcomes(self):
        plan = {"a": 1, "b": 2}
        run = ux_helpers.partial_verification_outcome(ux_helpers.VERIFY_RUN, plan)
        self.assertEqual(run.run_plan, plan)
        self.assertIsNot(run.run_plan, plan)
        self.assertFalse(run.save_plan or run.keep_verifying)

        save = ux_helpers.partial_verification_outcome(ux_helpers.VERIFY_SAVE, plan)
        self.assertEqual(save.run_plan, {})
        self.assertTrue(save.save_plan)

        keep = ux_helpers.partial_verification_outcome(ux_helpers.VERIFY_CONTINUE, plan)
        self.assertTrue(keep.keep_verifying)
        self.assertEqual(keep.run_plan, {})

        for choice in (ux_helpers.VERIFY_DISCARD, None):
            gone = ux_helpers.partial_verification_outcome(choice, plan)
            self.assertEqual(gone.run_plan, {})
            self.assertFalse(gone.save_plan or gone.keep_verifying)


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class SetupProgressTests(unittest.TestCase):
    def test_step_order_per_paradigm_kind(self):
        self.assertEqual(wizard_steps(KIND_TST), (STEP_MICE, STEP_ROI, STEP_TST_TUNER))
        self.assertEqual(
            wizard_steps(KIND_FREESTYLE), (STEP_ZONES, STEP_ROI, STEP_DETECTION)
        )
        self.assertEqual(
            wizard_steps(KIND_ARMS), (STEP_ROI, STEP_ARM_RULES, STEP_DETECTION)
        )
        self.assertEqual(wizard_steps(KIND_STANDARD), (STEP_ROI, STEP_DETECTION))

    def test_progress_tracks_completed_steps_and_next_step(self):
        progress = SetupProgress("Y-Maze", Path("a.mp4"), KIND_ARMS)
        self.assertEqual(progress.next_step(), STEP_ROI)
        self.assertFalse(progress.is_resumable)

        progress.roi_manager = ROIManager(source_dimensions=(100, 80))
        progress.mark(STEP_ROI)
        progress.mark(STEP_ROI)  # marking twice is harmless
        self.assertEqual(progress.completed, [STEP_ROI])
        self.assertEqual(progress.next_step(), STEP_ARM_RULES)
        self.assertTrue(progress.is_resumable)
        self.assertIn("regions of interest", progress.describe())
        self.assertIn("arm-entry rules", progress.describe())

        progress.mark(STEP_ARM_RULES, {"min_dwell_seconds": 0.5})
        progress.mark(STEP_DETECTION)
        self.assertIsNone(progress.next_step())
        self.assertEqual(progress.algorithm_parameters, {"min_dwell_seconds": 0.5})
        self.assertIn("final review", progress.describe())

    def test_progress_applies_only_to_the_same_paradigm_and_video_size(self):
        progress = SetupProgress("Y-Maze", Path("a.mp4"), KIND_ARMS)
        self.assertFalse(progress.applies_to("Y-Maze", (100, 80)))
        progress.roi_manager = ROIManager(source_dimensions=(100, 80))
        self.assertTrue(progress.applies_to("Y-Maze", (100, 80)))
        self.assertFalse(progress.applies_to("Elevated Plus Maze", (100, 80)))
        self.assertFalse(progress.applies_to("Y-Maze", (640, 480)))
        self.assertTrue(progress.applies_to("Y-Maze", None))


# ----------------------------------------------------------------------
# Dialog guards
# ----------------------------------------------------------------------
class _Maze:
    def get_roi_definitions(self):
        return [("arm_a", "Arm A")]


class _Capture:
    def get(self, _property):
        return 20

    def set(self, _property, _value):
        return True

    def read(self):
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        pass


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class DiscardGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _confirm(self, answer):
        return patch.object(ux_helpers, "confirm_discard", return_value=answer)

    # -- ROI selector ---------------------------------------------------
    def _roi_dialog(self):
        with patch("gui.roi_selector.cv2.VideoCapture", return_value=_Capture()):
            dialog = RoiSelectorDialog("example.mp4", _Maze())
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_roi_selector_closes_without_asking_when_nothing_was_drawn(self):
        dialog = self._roi_dialog()
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_not_called()
        self.assertEqual(dialog.result(), 0)

    def test_roi_selector_asks_before_discarding_a_drawn_polygon(self):
        dialog = self._roi_dialog()
        dialog.current_polygon_points = [QPointF(1, 1), QPointF(30, 1), QPointF(30, 30)]
        dialog.finalize_polygon()
        dialog.setResult(7)
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_called_once()
        self.assertEqual(dialog.result(), 7, "declining must keep the dialog open")

        dialog.show()
        with self._confirm(True):
            dialog.reject()
        self.assertFalse(dialog.isVisible())

    def test_roi_selector_counts_unfinished_points_as_work(self):
        dialog = self._roi_dialog()
        dialog.current_polygon_points = [QPointF(5, 5)]
        self.assertTrue(dialog.has_unsaved_work())

    def test_roi_selector_escape_goes_through_the_same_guard(self):
        dialog = self._roi_dialog()
        dialog.current_polygon_points = [QPointF(5, 5)]
        dialog.show()
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        with self._confirm(False) as confirm:
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        confirm.assert_called_once()
        self.assertTrue(dialog.isVisible())
        dialog.current_polygon_points = []  # nothing left to confirm on close
        dialog.close()

    # -- Freestyle config -------------------------------------------------
    def test_freestyle_dialog_asks_only_after_a_change(self):
        dialog = FreestyleConfigDialog()
        self.addCleanup(dialog.deleteLater)
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_not_called()

        dialog.zone_name_inputs[0].setText("Centre")
        dialog.setResult(7)
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_called_once()
        self.assertEqual(dialog.result(), 7)

        dialog.immobility_threshold_spin.setValue(9.0)
        with self._confirm(True):
            dialog.reject()
        self.assertEqual(dialog.result(), 0)

    # -- Settings dialog --------------------------------------------------
    def _settings_dialog(self, is_verification):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        video = support.write_tracking_video(Path(tmp.name), frames=30)
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square())
        settings = BatchSettings(manager, AppConfig().detection, 0.0)
        dialog = SettingsDialog(
            str(video), settings, is_verification=is_verification,
            maze_name="Y-Maze",
        )
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_settings_review_asks_once_edited(self):
        dialog = self._settings_dialog(is_verification=False)
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_not_called()

        dialog.timelapse_check.setChecked(True)
        dialog.setResult(7)
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_called_once()
        self.assertEqual(dialog.result(), 7)
        with self._confirm(True):
            dialog.reject()
        self.assertEqual(dialog.result(), 0)

    def test_verification_dialog_leaves_the_decision_to_the_main_window(self):
        dialog = self._settings_dialog(is_verification=True)
        dialog.timelapse_check.setChecked(True)
        with self._confirm(False) as confirm:
            dialog.reject()
        confirm.assert_not_called()

    # -- Detection tuner -----------------------------------------------------
    def test_tuner_has_ok_and_cancel_and_guards_moved_sliders(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        video = support.write_tracking_video(Path(tmp.name), frames=12)
        ref = np.full((120, 160), 40, dtype=np.uint8)
        dialog = TunerDialog(str(video), ref, ROIManager(source_dimensions=(160, 120)))
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.accept_button.text(), "OK")
        self.assertEqual(dialog.cancel_button.text(), "Cancel")

        with self._confirm(False) as confirm:
            dialog.cancel_button.click()
        confirm.assert_not_called()

        dialog = TunerDialog(str(video), ref, ROIManager(source_dimensions=(160, 120)))
        self.addCleanup(dialog.deleteLater)
        dialog.sliders["Window Size"].setValue(80)
        dialog.setResult(7)
        with self._confirm(False) as confirm:
            dialog.cancel_button.click()
        confirm.assert_called_once()
        self.assertEqual(dialog.result(), 7)
        with self._confirm(True):
            dialog.cancel_button.click()
        self.assertEqual(dialog.result(), 0)

    # -- TST tuner -----------------------------------------------------------
    def test_tst_tuner_asks_only_after_the_scan_finished(self):
        roi_manager = ROIManager(source_dimensions=(100, 100))
        roi_manager.add_roi("mouse_1", _square())
        tst = create_maze("Tail Suspension Test")
        with patch.object(TstTunerDialog, "start_scan"):
            dialog = TstTunerDialog(Path("unopened.mp4"), roi_manager, tst)
        self.addCleanup(dialog.deleteLater)
        try:
            with self._confirm(False) as confirm:
                dialog.reject()
            confirm.assert_not_called()

            dialog.accept_button.setEnabled(True)  # what a finished scan does
            dialog.setResult(7)
            with self._confirm(False) as confirm:
                dialog.reject()
            confirm.assert_called_once()
            self.assertEqual(dialog.result(), 7)
        finally:
            dialog.cleanup()


# ----------------------------------------------------------------------
# Main window: wizard progress and verification
# ----------------------------------------------------------------------
class _FakeDialog:
    """Stands in for a modal dialog: records construction, returns a result."""

    def __init__(self, result=True, **attributes):
        self.result = result
        self.created = 0
        self.attributes = attributes

    def __call__(self, *args, **kwargs):
        self.created += 1
        outer = self

        class _Instance:
            def exec(self_inner):
                return outer.result

            def deleteLater(self_inner):
                pass

            def __getattr__(self_inner, name):
                value = outer.attributes[name]
                return value if callable(value) else (lambda: value)

        return _Instance()


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class WizardProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = support.write_tracking_video(self.root, frames=20)
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        self.window.video_paths = [self.video]

        self.roi_manager = ROIManager(source_dimensions=(160, 120))
        self.roi_manager.add_roi("arm_a", _square())
        self.roi = _FakeDialog(True, get_roi_manager=self.roi_manager)
        self.arm = _FakeDialog(
            True, get_parameters={"min_dwell_seconds": 0.7, "count_initial_arm": True}
        )
        self.tuner = _FakeDialog(True, get_detection_config=AppConfig().detection)
        for name, fake in (
            ("RoiSelectorDialog", self.roi),
            ("ArmEntryParametersDialog", self.arm),
            ("TunerDialog", self.tuner),
        ):
            patcher = patch(f"gui.main_window.{name}", fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        ref = patch.object(
            self.window, "_generate_reference_frame",
            return_value=np.zeros((120, 160), dtype=np.uint8),
        )
        ref.start()
        self.addCleanup(ref.stop)

    def _wizard(self, progress=None):
        return self.window._run_setup_wizard(self.video, "Y-Maze", progress)

    def test_cancelling_before_the_roi_step_keeps_nothing(self):
        self.roi.result = False
        self.assertIsNone(self._wizard())
        self.assertIsNone(self.window._pending_setup)

    def test_cancelling_after_the_roi_step_keeps_the_rois_and_rules(self):
        self.tuner.result = False
        self.assertIsNone(self._wizard())

        pending = self.window._pending_setup
        self.assertIsNotNone(pending)
        self.assertIs(pending.roi_manager, self.roi_manager)
        self.assertEqual(pending.completed, [STEP_ROI, STEP_ARM_RULES])
        self.assertEqual(pending.algorithm_parameters["min_dwell_seconds"], 0.7)
        self.assertIsNone(pending.base_settings)

    def test_resuming_skips_finished_steps(self):
        self.tuner.result = False
        self._wizard()
        pending = self.window._pending_setup
        self.roi.created = self.arm.created = 0

        self.tuner.result = True
        settings = self._wizard(pending)

        self.assertIsNotNone(settings)
        self.assertEqual((self.roi.created, self.arm.created), (0, 0))
        self.assertIs(settings.roi_manager, self.roi_manager)
        self.assertEqual(pending.completed[-1], STEP_DETECTION)
        # Only the final review is left, so it stays pending until accepted.
        self.assertIs(self.window._pending_setup.base_settings, settings)

    def test_a_cancelled_review_can_be_resumed_straight_at_the_review(self):
        self._wizard()
        pending = self.window._pending_setup
        self.roi.created = self.tuner.created = 0
        again = self._wizard(pending)
        self.assertIs(again, pending.base_settings)
        self.assertEqual((self.roi.created, self.tuner.created), (0, 0))

    def test_offer_resume_setup_choices(self):
        self.tuner.result = False
        self._wizard()
        pending = self.window._pending_setup

        def offer(answer):
            with patch.object(ux_helpers, "ask_choice", return_value=answer):
                return self.window._offer_resume_setup(self.video, "Y-Maze")

        self.assertIs(offer("resume"), pending)
        self.assertEqual(self.window.maze.min_dwell_seconds, 0.7)

        from gui import main_window
        self.assertIs(offer("cancel"), main_window._CANCEL)
        self.assertIs(self.window._pending_setup, pending)

        self.assertIsNone(offer("restart"))
        self.assertIsNone(self.window._pending_setup)

    def test_no_offer_for_another_paradigm_or_without_rois(self):
        with patch.object(ux_helpers, "ask_choice") as ask:
            self.assertIsNone(self.window._offer_resume_setup(self.video, "Y-Maze"))
            self.tuner.result = False
            self._wizard()
            self.assertIsNone(
                self.window._offer_resume_setup(self.video, "Elevated Plus Maze")
            )
        ask.assert_not_called()

    def test_accepted_review_clears_the_pending_setup(self):
        settings = self._wizard()
        self.assertIsNotNone(self.window._pending_setup)
        review = _FakeDialog(True, get_settings=settings)
        with patch("gui.main_window.SettingsDialog", review):
            result = self.window._review_settings(self.video, settings)
        self.assertIs(result, settings)
        self.assertIsNone(self.window._pending_setup)
        self.assertIs(self.window.current_settings, settings)

    def test_cancelled_review_keeps_the_pending_setup(self):
        settings = self._wizard()
        review = _FakeDialog(False)
        with patch("gui.main_window.SettingsDialog", review):
            self.assertIsNone(self.window._review_settings(self.video, settings))
        self.assertIsNotNone(self.window._pending_setup)


class _ScriptedVerifyDialog:
    """SettingsDialog stand-in that replays (exec result, action) pairs."""

    script = []
    seen = []

    def __init__(self, video_path, settings, **_kwargs):
        self.video_path = video_path
        self.settings = settings
        _ScriptedVerifyDialog.seen.append(Path(video_path).name)
        self.result_action = "cancel"

    def setWindowTitle(self, _title):
        pass

    def exec(self):
        accepted, action = _ScriptedVerifyDialog.script.pop(0)
        self.result_action = action
        return accepted

    def get_settings(self):
        return self.settings

    def get_real_world_cm(self):
        return 0.0

    def deleteLater(self):
        pass


@unittest.skipUnless(HAS_RUNTIME, "GUI runtime is not installed")
class VerificationCancelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(temp_app_settings(self))
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window._detach_log_handler)
        self.window.maze = create_maze("Y-Maze")
        self.videos = [Path(f"v{i}.mp4") for i in range(4)]
        self.window.video_paths = list(self.videos)
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", _square())
        self.base = BatchSettings(manager, AppConfig().detection, 0.0)
        _ScriptedVerifyDialog.seen = []
        patcher = patch("gui.main_window.SettingsDialog", _ScriptedVerifyDialog)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _verify(self, script, choice):
        _ScriptedVerifyDialog.script = list(script)
        with patch.object(ux_helpers, "ask_choice", return_value=choice) as ask:
            plan = self.window.build_analysis_plan(self.base, verify_each=True)
        return plan, ask

    def test_cancelling_with_nothing_accepted_asks_nothing(self):
        plan, ask = self._verify([(False, "cancel")], ux_helpers.VERIFY_DISCARD)
        self.assertEqual(plan, {})
        ask.assert_not_called()

    def test_run_the_verified_videos(self):
        plan, ask = self._verify(
            [(True, "accept"), (True, "accept"), (False, "cancel")],
            ux_helpers.VERIFY_RUN,
        )
        self.assertEqual(list(plan), self.videos[:2])
        labels = [c.label for c in ask.call_args.args[3]]
        self.assertIn("Run the 2 verified videos", labels)

    def test_discard_runs_nothing(self):
        plan, _ = self._verify(
            [(True, "accept"), (False, "cancel")], ux_helpers.VERIFY_DISCARD
        )
        self.assertEqual(plan, {})

    def test_save_partial_plan_saves_only_accepted_videos_and_runs_nothing(self):
        script = [(True, "accept"), (True, "accept"), (False, "cancel")]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            target = Path(tmp) / "partial.json"
            with patch(
                "gui.main_window.QFileDialog.getSaveFileName",
                return_value=(str(target), ""),
            ):
                plan, _ = self._verify(script, ux_helpers.VERIFY_SAVE)
            saved = BatchSettings.load_batch_plan(target)
        self.assertEqual(plan, {})
        self.assertEqual(len(saved), 2)
        self.assertEqual(
            sorted(Path(key).name for key in saved), ["v0.mp4", "v1.mp4"]
        )

    def test_keep_verifying_shows_the_same_video_again(self):
        script = [(True, "accept"), (False, "cancel"), (True, "accept"),
                  (True, "skip"), (True, "skip")]
        plan, _ = self._verify(script, ux_helpers.VERIFY_CONTINUE)
        self.assertEqual(
            _ScriptedVerifyDialog.seen, ["v0.mp4", "v1.mp4", "v1.mp4", "v2.mp4", "v3.mp4"]
        )
        self.assertEqual(list(plan), self.videos[:2])


if __name__ == "__main__":
    unittest.main()
