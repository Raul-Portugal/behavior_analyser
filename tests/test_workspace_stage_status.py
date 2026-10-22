"""Stepper statuses from the project alone (the Qt-free stage rules)."""
import unittest

import numpy as np

from core.models import AppConfig, BatchSettings
from core.run_outputs import write_run_plan
from core.project import START_PER_VIDEO, Project, TrialProtocol
from gui.workspace.stages import status as st
from mazes import AVAILABLE_MAZES, create_maze
from roi_manager import ROIManager
from tests.project_support import ProjectFixture


def settings_for(paradigm: str, size=(640, 480)) -> BatchSettings:
    """Valid settings of a real paradigm (every required region drawn)."""
    manager = ROIManager(source_dimensions=size)
    for index, (name, _) in enumerate(create_maze(paradigm).get_roi_definitions()):
        x = index * 100
        manager.add_roi(name, np.array([[x, 0], [x + 40, 0], [x + 40, 40], [x, 40]]))
    return BatchSettings(
        manager, AppConfig().detection, 0.0, paradigm_id=paradigm,
        algorithm_parameters=create_maze(paradigm).get_algorithm_parameters(),
    )


class StageStatusTests(ProjectFixture):
    def make_project(self, paradigm="Y-Maze", videos=2, template=True):
        project = Project(name="Study", paradigm=paradigm)
        if template:
            project.set_template(settings_for(paradigm))
        project.add_videos([
            self.write_video(f"v{n}.mp4", b"x" * (n + 1)) for n in range(videos)
        ])
        return project

    def status(self, stage, project):
        return st.stage_status(stage, project)

    # -- no project ---------------------------------------------------------
    def test_without_a_project_only_videos_is_available(self):
        statuses = st.all_stage_statuses(None)
        self.assertEqual(statuses[st.STAGE_VIDEOS].status, st.STATUS_TODO)
        for stage in st.STAGE_ORDER[1:]:
            with self.subTest(stage):
                self.assertEqual(statuses[stage].status, st.STATUS_DISABLED)
                self.assertIn("project", statuses[stage].reason)

    def test_order_matches_the_flow(self):
        self.assertEqual(
            st.STAGE_ORDER,
            ("videos", "arena", "detection", "trials", "review", "results"),
        )

    # -- videos -------------------------------------------------------------
    def test_videos_needs_a_paradigm(self):
        project = Project(name="Study")
        found = self.status(st.STAGE_VIDEOS, project)
        self.assertEqual(found.status, st.STATUS_TODO)
        self.assertIn("test", found.reason)

    def test_videos_needs_a_video(self):
        project = Project(name="Study", paradigm="Y-Maze")
        self.assertEqual(self.status(st.STAGE_VIDEOS, project).status, st.STATUS_TODO)

    def test_videos_done_with_a_paradigm_and_a_video(self):
        project = self.make_project(videos=3, template=False)
        found = self.status(st.STAGE_VIDEOS, project)
        self.assertEqual(found.status, st.STATUS_DONE)
        self.assertIn("3 videos", found.reason)

    def test_videos_singular(self):
        project = self.make_project(videos=1, template=False)
        self.assertIn("1 video ", self.status(st.STAGE_VIDEOS, project).reason)

    def test_a_missing_file_needs_attention(self):
        project = self.make_project(template=False)
        project.videos[0].path.unlink()
        found = self.status(st.STAGE_VIDEOS, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)
        self.assertIn("1 video", found.reason)

    def test_later_stages_wait_for_paradigm_and_video(self):
        for project, reason in (
            (Project(name="S"), st.NEEDS_PARADIGM),
            (Project(name="S", paradigm="Y-Maze"), st.NEEDS_VIDEO),
        ):
            for stage in (st.STAGE_ARENA, st.STAGE_DETECTION, st.STAGE_TRIALS,
                          st.STAGE_REVIEW):
                with self.subTest(stage=stage, reason=reason):
                    found = self.status(stage, project)
                    self.assertEqual(found.status, st.STATUS_DISABLED)
                    self.assertEqual(found.reason, reason)

    def test_results_is_available_with_any_project(self):
        project = Project(name="S")
        self.assertEqual(self.status(st.STAGE_RESULTS, project).status, st.STATUS_TODO)

    # -- arena --------------------------------------------------------------
    def test_arena_is_todo_without_a_template(self):
        project = self.make_project(template=False)
        self.assertEqual(self.status(st.STAGE_ARENA, project).status, st.STATUS_TODO)

    def test_arena_done_with_valid_regions(self):
        project = self.make_project()
        self.assertEqual(self.status(st.STAGE_ARENA, project).status, st.STATUS_DONE)

    def test_arena_attention_for_invalid_regions(self):
        project = self.make_project(template=False)
        template = settings_for("Y-Maze")
        template.roi_manager = ROIManager(source_dimensions=(640, 480))
        template.roi_manager.add_roi(
            "arm_a", np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        )
        project.set_template(template)
        found = self.status(st.STAGE_ARENA, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)
        self.assertTrue(found.reason)

    def test_arena_attention_when_a_video_needs_alignment(self):
        project = self.make_project()
        project.set_video_info(project.videos[0].key, frame_size=(1280, 720))
        found = self.status(st.STAGE_ARENA, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)
        self.assertIn("1 video need", found.reason)
        # The same video matching the drawn frame size is fine again.
        project.set_video_info(project.videos[0].key, frame_size=(640, 480))
        self.assertEqual(self.status(st.STAGE_ARENA, project).status, st.STATUS_DONE)

    def test_arena_uses_the_issues_it_is_given(self):
        project = self.make_project()
        project.set_video_info(project.videos[0].key, frame_size=(1280, 720))
        found = st.arena_status(project, issues=[])
        self.assertEqual(found.status, st.STATUS_DONE)

    # -- detection ----------------------------------------------------------
    def test_detection_todo_before_the_template_exists(self):
        project = self.make_project(template=False)
        self.assertEqual(self.status(st.STAGE_DETECTION, project).status, st.STATUS_TODO)

    def test_detection_done_once_reviewed(self):
        project = self.make_project()
        self.assertEqual(self.status(st.STAGE_DETECTION, project).status, st.STATUS_TODO)
        project.set_detection_reviewed()
        self.assertEqual(self.status(st.STAGE_DETECTION, project).status, st.STATUS_DONE)

    def test_detection_attention_for_an_invalid_configuration(self):
        project = self.make_project()
        project.template_settings.detection_config.threshold_percentile = 150.0
        found = self.status(st.STAGE_DETECTION, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)

    def test_the_tst_detection_stage_is_motion_tuning(self):
        self.assertEqual(st.detection_title("Tail Suspension Test"), "Motion tuning")
        self.assertEqual(st.detection_title("Y-Maze"), "Detection")
        self.assertEqual(st.detection_title(""), "Detection")
        project = self.make_project(paradigm="Tail Suspension Test")
        project.set_detection_reviewed()
        self.assertEqual(self.status(st.STAGE_DETECTION, project).status, st.STATUS_DONE)

    def test_every_paradigm_has_a_detection_step(self):
        for paradigm in AVAILABLE_MAZES:
            with self.subTest(paradigm):
                self.assertIn(st.detection_step(paradigm),
                              ("detection_tuning", "tst_tuning"))

    # -- trials -------------------------------------------------------------
    def test_trials_todo_before_the_template_exists(self):
        project = self.make_project(template=False)
        self.assertEqual(self.status(st.STAGE_TRIALS, project).status, st.STATUS_TODO)

    def test_trials_done_when_every_trial_is_ok(self):
        project = self.make_project()
        self.assertEqual(self.status(st.STAGE_TRIALS, project).status, st.STATUS_DONE)

    def test_trials_attention_for_videos_without_a_start(self):
        project = self.make_project()
        project.set_trial_protocol(300.0, START_PER_VIDEO)
        found = self.status(st.STAGE_TRIALS, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)
        self.assertIn("2 videos need a start", found.reason)
        project.set_trial(project.videos[0].key, start=12.0)
        project.set_trial(project.videos[1].key, start=20.0)
        self.assertEqual(self.status(st.STAGE_TRIALS, project).status, st.STATUS_DONE)

    def test_trials_attention_for_invalid_times(self):
        project = self.make_project()
        project.set_video_info(project.videos[0].key, duration_s=60.0)
        project.set_trial(project.videos[0].key, start=90.0)
        found = self.status(st.STAGE_TRIALS, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)
        self.assertIn("1 video have", found.reason)

    # -- review -------------------------------------------------------------
    def test_review_todo_when_nothing_is_configured(self):
        project = self.make_project(template=False)
        self.assertEqual(self.status(st.STAGE_REVIEW, project).status, st.STATUS_TODO)

    def test_review_done_when_ready_to_run(self):
        project = self.make_project()
        found = self.status(st.STAGE_REVIEW, project)
        self.assertEqual(found.status, st.STATUS_DONE)
        self.assertEqual(found.reason, "Ready to run")

    def test_review_attention_when_an_error_blocks_the_run(self):
        project = self.make_project()
        project.videos[1].path.unlink()
        found = self.status(st.STAGE_REVIEW, project)
        self.assertEqual(found.status, st.STATUS_ATTENTION)
        self.assertIn("1 video need", found.reason)

    def test_review_todo_when_some_videos_have_no_settings(self):
        project = self.make_project()
        project.add_videos([self.write_video("late.mp4", b"late")])
        found = self.status(st.STAGE_REVIEW, project)
        # The new video inherits the template, so it is configured too.
        self.assertEqual(found.status, st.STATUS_DONE)
        project.videos[-1].settings = None
        self.assertEqual(self.status(st.STAGE_REVIEW, project).status, st.STATUS_TODO)

    # -- results ------------------------------------------------------------
    def test_results_done_when_a_run_exists(self):
        project = self.make_project()
        self.assertEqual(self.status(st.STAGE_RESULTS, project).status, st.STATUS_TODO)
        run_dir = self.root / "analysis_run"
        run_dir.mkdir()
        write_run_plan({v.path: settings_for("Y-Maze") for v in project.videos}, run_dir)
        project.record_run(run_dir)
        found = self.status(st.STAGE_RESULTS, project)
        self.assertEqual(found.status, st.STATUS_DONE)
        self.assertIn("1 run", found.reason)

    # -- all ----------------------------------------------------------------
    def test_all_statuses_cover_every_stage_once(self):
        project = self.make_project()
        statuses = st.all_stage_statuses(project)
        self.assertEqual(tuple(statuses), st.STAGE_ORDER)
        self.assertTrue(all(isinstance(s, st.StageStatus) for s in statuses.values()))

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(KeyError):
            st.stage_status("nope", None)

    def test_protocol_object_is_unchanged_by_status_checks(self):
        project = self.make_project()
        before = project.trial_protocol.to_dict()
        st.all_stage_statuses(project)
        self.assertEqual(project.trial_protocol.to_dict(), before)
        self.assertIsInstance(project.trial_protocol, TrialProtocol)


if __name__ == "__main__":
    unittest.main()
