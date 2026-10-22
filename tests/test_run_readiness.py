"""The readiness checklist of Review & run: checks, problem groups, blockers."""
import unittest

from core.project import START_PER_VIDEO, Project
from gui.workspace import run_readiness as rr
from gui.workspace.run_check import SCOPE_ALL, SCOPE_PENDING, SCOPE_SELECTED
from gui.workspace.stages import status as st
from tests.project_support import ProjectFixture
from tests.test_workspace_stage_status import settings_for


class ReadinessFixture(ProjectFixture):
    def make_project(self, videos=3, template=True, paradigm="Y-Maze"):
        project = Project(name="Study", paradigm=paradigm)
        if template:
            project.set_template(settings_for(paradigm))
        project.add_videos([
            self.write_video(f"v{n}.mp4", b"x" * (n + 1)) for n in range(videos)
        ])
        return project

    def readiness(self, project, scope=SCOPE_ALL, keys=None):
        keys = project.keys() if keys is None else keys
        return rr.build_readiness(project, scope, keys)


class ChecklistTests(ReadinessFixture):
    def test_one_row_per_stage_in_order(self):
        found = self.readiness(self.make_project())
        self.assertEqual(
            [c.stage_id for c in found.checks],
            [st.STAGE_VIDEOS, st.STAGE_ARENA, st.STAGE_DETECTION, st.STAGE_TRIALS],
        )
        self.assertEqual(
            [c.title for c in found.checks],
            ["Videos", "Arena & zones", "Detection", "Trials"],
        )

    def test_the_motion_stage_has_its_own_title(self):
        project = self.make_project(paradigm="Tail Suspension Test")
        self.assertEqual(self.readiness(project).checks[2].title, "Motion tuning")

    def test_done_rows_have_nothing_to_fix(self):
        found = self.readiness(self.make_project())
        videos = found.check(st.STAGE_VIDEOS)
        self.assertEqual((videos.chip, videos.label, videos.fix_stage),
                         (rr.CHIP_OK, "Done", ""))
        self.assertTrue(videos.done)

    def test_an_unreviewed_detection_is_a_warning_that_does_not_block(self):
        project = self.make_project()
        found = self.readiness(project)
        detection = found.check(st.STAGE_DETECTION)
        self.assertEqual(detection.chip, rr.CHIP_WARNING)
        self.assertEqual(detection.label, "Not reviewed")
        self.assertEqual(detection.fix_stage, st.STAGE_DETECTION)
        self.assertTrue(found.ready)
        project.set_detection_reviewed(True)
        self.assertEqual(self.readiness(project).check(st.STAGE_DETECTION).chip,
                         rr.CHIP_OK)

    def test_trials_that_need_a_start_are_an_error_with_a_fix(self):
        project = self.make_project()
        project.set_trial_protocol(300.0, START_PER_VIDEO)
        trials = self.readiness(project).check(st.STAGE_TRIALS)
        self.assertEqual(trials.chip, rr.CHIP_ERROR)
        self.assertEqual(trials.fix_stage, st.STAGE_TRIALS)
        self.assertIn("need a start time", trials.reason)

    def test_without_a_project_everything_waits(self):
        found = rr.build_readiness(None, SCOPE_ALL, [])
        self.assertFalse(found.ready)
        self.assertEqual(found.checks, [])
        self.assertIn("project", found.why_not)

    def test_a_locked_stage_points_back_to_the_videos(self):
        project = Project(name="Empty", paradigm="Y-Maze")
        found = rr.build_readiness(project, SCOPE_ALL, [])
        arena = found.check(st.STAGE_ARENA)
        self.assertEqual((arena.chip, arena.label), (rr.CHIP_NEUTRAL, "Waiting"))
        self.assertEqual(arena.fix_stage, st.STAGE_VIDEOS)


class BlockerTests(ReadinessFixture):
    def test_a_ready_project_has_no_blockers(self):
        found = self.readiness(self.make_project())
        self.assertTrue(found.ready)
        self.assertEqual(found.why_not, "")
        self.assertEqual((found.scope_count, found.configured_count, found.left_out),
                         (3, 3, 0))

    def test_no_videos_blocks(self):
        found = self.readiness(self.make_project(videos=0))
        self.assertFalse(found.ready)
        self.assertIn("Add the videos", found.why_not)

    def test_no_paradigm_blocks(self):
        project = Project(name="No test")
        project.add_videos([self.write_video("v.mp4")])
        found = self.readiness(project)
        self.assertIn("Choose the test", found.why_not)

    def test_videos_without_settings_block_when_none_has_any(self):
        found = self.readiness(self.make_project(template=False))
        self.assertFalse(found.ready)
        self.assertIn("None of the videos has settings", found.why_not)

    def test_a_video_without_settings_is_left_out_with_a_warning(self):
        project = self.make_project()
        project.entry(project.keys()[2]).settings = None
        found = self.readiness(project)
        self.assertTrue(found.ready)
        self.assertEqual((found.configured_count, found.left_out), (2, 1))
        group = found.group("unconfigured")
        self.assertEqual(group.kind, "warning")
        self.assertFalse(group.blocking)
        self.assertEqual(found.warnings, [group.summary()])

    def test_a_selected_video_without_settings_blocks(self):
        project = self.make_project()
        key = project.keys()[2]
        project.entry(key).settings = None
        found = self.readiness(project, SCOPE_SELECTED, [key])
        self.assertFalse(found.ready)
        self.assertTrue(found.group("unconfigured").blocking)

    def test_nothing_selected_asks_for_a_selection(self):
        found = self.readiness(self.make_project(), SCOPE_SELECTED, [])
        self.assertFalse(found.ready)
        self.assertIn("Select the videos", found.why_not)

    def test_a_missing_file_blocks_and_names_the_video(self):
        project = self.make_project()
        project.videos[1].path.unlink()
        found = self.readiness(project)
        group = found.group("missing_file")
        self.assertTrue(group.blocking)
        self.assertEqual(group.fix_stage, st.STAGE_VIDEOS)
        self.assertEqual(group.names, ("v1.mp4",))
        self.assertIn("Video files not found: 1 video", found.why_not)

    def test_missing_starts_block_and_are_grouped_per_kind(self):
        project = self.make_project()
        project.set_trial_protocol(300.0, START_PER_VIDEO)
        found = self.readiness(project)
        group = found.group("trial_needs_start")
        self.assertEqual(group.count, 3)
        self.assertEqual(group.fix_stage, st.STAGE_TRIALS)
        self.assertEqual(group.summary(), "Start time needed: 3 videos")
        self.assertIn("v0.mp4", group.listed_names())
        self.assertFalse(found.ready)

    def test_only_the_videos_of_the_scope_count(self):
        project = self.make_project()
        project.set_trial_protocol(300.0, START_PER_VIDEO)
        keys = project.keys()
        project.set_trial(keys[0], start=2.0)
        found = self.readiness(project, SCOPE_SELECTED, [keys[0]])
        self.assertTrue(found.ready)
        self.assertIsNone(found.group("trial_needs_start"))
        everything = self.readiness(project)
        self.assertEqual(everything.group("trial_needs_start").count, 2)

    def test_several_problems_are_counted_in_the_sentence(self):
        project = self.make_project()
        project.set_trial_protocol(300.0, START_PER_VIDEO)
        project.videos[0].path.unlink()
        found = self.readiness(project)
        self.assertGreaterEqual(len(found.blockers), 2)
        self.assertIn("more problem", found.why_not)

    def test_the_not_yet_done_scope_uses_the_keys_it_is_given(self):
        project = self.make_project()
        found = self.readiness(project, SCOPE_PENDING, project.keys()[:2])
        self.assertEqual(found.scope_count, 2)

    def test_unknown_keys_are_ignored(self):
        found = self.readiness(self.make_project(), SCOPE_ALL, ["nope"])
        self.assertEqual(found.scope_count, 0)
        self.assertFalse(found.ready)

    def test_a_zone_size_mismatch_blocks_and_points_to_the_arena(self):
        project = self.make_project()
        project.set_video_info(project.keys()[0], 10.0, (1280, 720))
        group = self.readiness(project).group("zones_size_mismatch")
        self.assertTrue(group.blocking)
        self.assertEqual(group.fix_stage, st.STAGE_ARENA)


class GroupTests(unittest.TestCase):
    def test_names_are_listed_with_a_count_of_the_rest(self):
        names = tuple(f"v{n}.mp4" for n in range(9))
        group = rr.ProblemGroup("x", "Title", "error", "", tuple(map(str, range(9))), names)
        self.assertTrue(group.listed_names().endswith("and 3 more"))

    def test_a_project_level_note_has_no_video_count(self):
        group = rr.ProblemGroup("sample_sheet_missing", "Sample sheet not found",
                                "warning", "", (), (), ("Sample sheet not found: x",))
        self.assertEqual(group.summary(), "Sample sheet not found")
        self.assertEqual(group.count, 1)


if __name__ == "__main__":
    unittest.main()
