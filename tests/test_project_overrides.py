"""Template + layered overrides: trial windows, zone overrides, the protocol.

The composed settings of a video must be identical (``to_dict`` and settings
hash) to the full copy that had the same values written into it.
"""
import unittest
from pathlib import Path

import numpy as np

from checkpoint_manager import CheckpointManager
from core.models import AppConfig, BatchSettings
from core.project import (START_PER_VIDEO, START_VIDEO, STATUS_CONFIGURED,
                          STATUS_DONE, TRIAL_INVALID, TRIAL_NEEDS_START,
                          TRIAL_OK, UNSET, ZONE_ADJUSTED, ZONE_ALIGNED,
                          ZONES_ADJUSTED, ZONES_ALIGNED, ZONES_CUSTOM,
                          ZONES_TEMPLATE, Project, ProjectError,
                          TrialProtocol, TrialWindow, ZoneOverride,
                          canonical_json, settings_hash)
from mazes import AVAILABLE_MAZES, create_maze
from roi_manager import ROIManager
from tests.project_support import ProjectFixture


def hash_of(settings: BatchSettings) -> str:
    return CheckpointManager(Path("."), create_dir=False).get_settings_hash(
        settings
    )


def paradigm_settings(paradigm: str, offset: int = 0, **kwargs) -> BatchSettings:
    """Settings of a real paradigm (its own parameters), cheap ROIs."""
    manager = ROIManager(source_dimensions=(640, 480))
    names = ["mouse_1", "mouse_2"] if paradigm == "Tail Suspension Test" else [
        "arm_a", "arm_b"
    ]
    for index, name in enumerate(names):
        x = offset + index * 100
        manager.add_roi(
            name, np.array([[x, 0], [x + 40, 0], [x + 40, 40], [x, 40]])
        )
    return BatchSettings(
        manager, AppConfig().detection, kwargs.pop("scale", 0.0),
        paradigm_id=paradigm,
        algorithm_parameters=create_maze(paradigm).get_algorithm_parameters(),
        **kwargs,
    )


def adjusted_zones(paradigm: str, shift: int = 7) -> ROIManager:
    return paradigm_settings(paradigm, offset=shift).roi_manager


class ComposedLikeFullCopy(ProjectFixture):
    """The headline rule, once per paradigm."""

    def _project(self, paradigm):
        project = Project(name="Study")
        project.set_template(paradigm_settings(paradigm))
        project.add_videos([
            self.write_video("a.mp4"), self.write_video("b.mp4", b"bb"),
        ])
        return project

    def assertSame(self, composed, manual):
        self.assertEqual(
            canonical_json(composed.to_dict()), canonical_json(manual.to_dict())
        )
        self.assertEqual(composed.to_dict(), manual.to_dict())
        self.assertEqual(hash_of(composed), hash_of(manual))
        self.assertEqual(settings_hash(composed), settings_hash(manual))

    def test_untouched_video_is_the_template(self):
        for paradigm in AVAILABLE_MAZES:
            with self.subTest(paradigm):
                project = self._project(paradigm)
                self.assertSame(
                    project.effective_settings(project.videos[0].key),
                    paradigm_settings(paradigm),
                )

    def test_trial_composes_like_a_full_copy(self):
        for paradigm in AVAILABLE_MAZES:
            with self.subTest(paradigm):
                project = self._project(paradigm)
                key = project.videos[0].key
                project.set_trial(key, start=12.5, end=312.5)
                manual = paradigm_settings(
                    paradigm, start_time=12.5, end_time=312.5
                )
                self.assertSame(project.effective_settings(key), manual)
                # The other video and the template are untouched.
                self.assertSame(
                    project.effective_settings(project.videos[1].key),
                    paradigm_settings(paradigm),
                )
                self.assertSame(
                    project.template_settings, paradigm_settings(paradigm)
                )

    def test_zone_override_composes_like_a_full_copy(self):
        for paradigm in AVAILABLE_MAZES:
            with self.subTest(paradigm):
                project = self._project(paradigm)
                key = project.videos[0].key
                project.set_zone_override(key, adjusted_zones(paradigm))
                self.assertSame(
                    project.effective_settings(key),
                    paradigm_settings(paradigm, offset=7),
                )

    def test_trial_and_zones_together_like_a_full_copy(self):
        for paradigm in AVAILABLE_MAZES:
            with self.subTest(paradigm):
                project = self._project(paradigm)
                key = project.videos[1].key
                project.set_trial(key, start=30.0, end=330.0)
                project.set_zone_override(
                    key, adjusted_zones(paradigm), source=ZONE_ADJUSTED,
                    scale_factor=0.25,
                )
                manual = paradigm_settings(
                    paradigm, offset=7, scale=0.25,
                    start_time=30.0, end_time=330.0,
                )
                self.assertSame(project.effective_settings(key), manual)

    def test_composition_with_a_template_that_has_its_own_times(self):
        project = Project()
        project.set_template(
            paradigm_settings("Y-Maze", start_time=5.0, end_time=600.0)
        )
        project.add_videos([self.write_video("a.mp4")])
        key = project.videos[0].key
        # Only the start is set: the end stays the template's.
        project.set_trial(key, start=20.0)
        self.assertSame(
            project.effective_settings(key),
            paradigm_settings("Y-Maze", start_time=20.0, end_time=600.0),
        )
        project.set_trial(key, start=UNSET)
        self.assertSame(
            project.effective_settings(key),
            paradigm_settings("Y-Maze", start_time=5.0, end_time=600.0),
        )

    def test_integer_times_keep_their_type_and_hash(self):
        project = self._project("Y-Maze")
        key = project.videos[0].key
        project.set_trial(key, start=12, end=312)
        manual = paradigm_settings("Y-Maze", start_time=12, end_time=312)
        self.assertSame(project.effective_settings(key), manual)

    def test_aligned_zones_rescale_the_calibration_like_an_alignment(self):
        from core import roi_alignment as ra
        template = paradigm_settings("Y-Maze")
        template.roi_manager.set_reference_line((0, 0), (100, 0), "ref")
        template.scale_factor = 0.5
        project = Project()
        project.set_template(template)
        project.add_videos([self.write_video("a.mp4")])
        matrix = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])
        aligned = ra.apply_alignment(template, matrix, (1280, 960))
        self.assertNotEqual(aligned.scale_factor, template.scale_factor)
        key = project.videos[0].key
        zones = project.set_zone_override(
            key, aligned.roi_manager, source=ZONE_ALIGNED, transform=matrix
        )
        self.assertEqual(zones.transform, matrix.tolist())
        self.assertEqual(zones.source, ZONE_ALIGNED)
        self.assertSame(project.effective_settings(key), aligned)

    def test_legacy_full_copy_with_a_trial_on_top(self):
        project = self._project("Y-Maze")
        key = project.videos[0].key
        legacy = paradigm_settings("Y-Maze", offset=3, scale=0.1)
        project.customize(key, legacy)
        self.assertSame(project.effective_settings(key), legacy)
        project.set_trial(key, start=9.0, end=99.0)
        self.assertSame(
            project.effective_settings(key),
            paradigm_settings(
                "Y-Maze", offset=3, scale=0.1, start_time=9.0, end_time=99.0
            ),
        )

    def test_effective_settings_is_a_fresh_copy(self):
        project = self._project("Y-Maze")
        key = project.videos[0].key
        first = project.effective_settings(key)
        first.start_time = 99.0
        first.roi_manager.rois.clear()
        self.assertEqual(project.effective_settings(key).start_time, 0.0)
        self.assertTrue(project.effective_settings(key).roi_manager.rois)

    def test_unconfigured_video_has_no_effective_settings(self):
        project = Project()
        project.add_videos([self.write_video("a.mp4")])
        self.assertIsNone(project.effective_settings(project.videos[0].key))

    def test_analysis_plan_carries_the_composed_settings(self):
        project = self._project("Y-Maze")
        key = project.videos[0].key
        project.set_trial(key, start=4.0, end=40.0)
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        plan = project.analysis_plan()
        first, second = project.videos
        self.assertSame(
            plan.plan[first.path],
            paradigm_settings("Y-Maze", offset=7, start_time=4.0, end_time=40.0),
        )
        self.assertSame(plan.plan[second.path], paradigm_settings("Y-Maze"))


class ProtocolTests(ProjectFixture):
    def _project(self, **protocol):
        project = Project()
        project.set_template(paradigm_settings("Elevated Plus Maze"))
        project.add_videos([
            self.write_video("a.mp4"), self.write_video("b.mp4", b"bb"),
        ])
        if protocol:
            project.set_trial_protocol(**protocol)
        return project

    def end_of(self, project, index):
        return project.effective_settings(project.videos[index].key).end_time

    def test_duration_applies_to_every_video_from_its_start(self):
        project = self._project(duration_s=300.0, start_mode=START_PER_VIDEO)
        project.set_trial(project.videos[0].key, start=12.0)
        self.assertEqual(self.end_of(project, 0), 312.0)
        # No trial: starts where the template does (0).
        self.assertEqual(self.end_of(project, 1), 300.0)

    def test_explicit_end_wins_over_the_protocol(self):
        project = self._project(duration_s=300.0)
        project.set_trial(project.videos[0].key, start=10.0, end=50.0)
        self.assertEqual(self.end_of(project, 0), 50.0)

    def test_protocol_like_a_full_copy(self):
        project = self._project(duration_s=300.0, start_mode=START_PER_VIDEO)
        key = project.videos[0].key
        project.set_trial(key, start=12.0)
        manual = paradigm_settings(
            "Elevated Plus Maze", start_time=12.0, end_time=312.0
        )
        self.assertEqual(
            project.effective_settings(key).to_dict(), manual.to_dict()
        )
        self.assertEqual(hash_of(project.effective_settings(key)), hash_of(manual))

    def test_end_is_capped_at_the_known_video_duration(self):
        project = self._project(duration_s=300.0, start_mode=START_PER_VIDEO)
        key = project.videos[0].key
        project.set_trial(key, start=100.0)
        self.assertEqual(self.end_of(project, 0), 400.0)
        project.set_video_info(key, duration_s=250.0)
        self.assertEqual(self.end_of(project, 0), 250.0)
        project.set_video_info(key, duration_s=1000.0)
        self.assertEqual(self.end_of(project, 0), 400.0)

    def test_start_beyond_the_video_is_invalid_not_capped_to_nonsense(self):
        project = self._project(duration_s=300.0, start_mode=START_PER_VIDEO)
        key = project.videos[0].key
        project.set_trial(key, start=100.0)
        project.set_video_info(key, duration_s=60.0)
        self.assertIsNone(self.end_of(project, 0))
        self.assertEqual(project.trial_status(key), TRIAL_INVALID)
        self.assertIn("beyond the end", project.trial_problem(key))

    def test_no_duration_means_until_the_video_ends(self):
        project = self._project(duration_s=None, start_mode=START_PER_VIDEO)
        project.set_trial(project.videos[0].key, start=12.0)
        self.assertIsNone(self.end_of(project, 0))

    def test_legacy_copy_keeps_its_end_until_it_has_a_trial(self):
        project = self._project(duration_s=300.0)
        key = project.videos[0].key
        project.customize(
            key, paradigm_settings("Elevated Plus Maze", offset=2, end_time=77.0)
        )
        self.assertEqual(self.end_of(project, 0), 77.0)
        project.set_trial(key, start=5.0)
        self.assertEqual(self.end_of(project, 0), 305.0)

    def test_set_protocol_reports_the_videos_it_changed(self):
        project = self._project()
        for entry in project.videos:
            entry.last_status = STATUS_DONE
        changed = project.set_trial_protocol(duration_s=120.0)
        self.assertEqual(changed, project.keys())
        self.assertTrue(
            all(v.last_status == STATUS_CONFIGURED for v in project.videos)
        )
        self.assertEqual(project.set_trial_protocol(duration_s=120.0), [])

    def test_protocol_validation(self):
        project = self._project()
        for bad in (0, -5, float("nan"), float("inf"), True):
            with self.assertRaises(ProjectError, msg=repr(bad)):
                project.set_trial_protocol(duration_s=bad)
        with self.assertRaises(ProjectError):
            project.set_trial_protocol(start_mode="sometimes")
        self.assertEqual(project.trial_protocol, TrialProtocol())

    def test_apply_protocol_to_all_drops_explicit_ends_and_video_starts(self):
        project = self._project(duration_s=300.0, start_mode=START_PER_VIDEO)
        a, b = (v.key for v in project.videos)
        project.set_trial(a, start=10.0, end=50.0)
        project.set_trial(b, start=20.0, end=60.0)
        self.assertEqual(project.apply_protocol_to_all(), [a, b])
        self.assertEqual(self.end_of(project, 0), 310.0)
        self.assertEqual(self.end_of(project, 1), 320.0)
        self.assertTrue(project.entry(a).trial.start_set)   # per-video start kept
        project.set_trial_protocol(duration_s=300.0, start_mode=START_VIDEO)
        project.apply_protocol_to_all()
        self.assertIsNone(project.entry(a).trial)
        self.assertEqual(self.end_of(project, 0), 300.0)


class TSTWindowTests(ProjectFixture):
    def _project(self):
        project = Project()
        project.set_template(paradigm_settings("Tail Suspension Test"))
        project.add_videos([
            self.write_video("a.mp4"), self.write_video("b.mp4", b"bb"),
        ])
        return project

    def test_per_mouse_windows_compose_like_a_full_copy(self):
        project = self._project()
        key = project.videos[0].key
        windows = {"mouse_1": (10.0, 250.0), "mouse_2": (25.0, 265.0)}
        project.set_trial(key, per_roi_times=windows)
        manual = paradigm_settings(
            "Tail Suspension Test", per_roi_times=windows
        )
        composed = project.effective_settings(key)
        self.assertEqual(composed.to_dict(), manual.to_dict())
        self.assertEqual(hash_of(composed), hash_of(manual))
        self.assertEqual(composed.per_roi_times, windows)
        # The other video has none.
        self.assertEqual(
            project.effective_settings(project.videos[1].key).per_roi_times, {}
        )

    def test_windows_replace_the_templates_and_can_be_cleared(self):
        project = Project()
        project.set_template(paradigm_settings(
            "Tail Suspension Test", per_roi_times={"mouse_1": (0.0, 360.0)}
        ))
        project.add_videos([self.write_video("a.mp4")])
        key = project.videos[0].key
        project.set_trial(key, per_roi_times={"mouse_1": (4.0, 100.0)})
        self.assertEqual(
            project.effective_settings(key).per_roi_times,
            {"mouse_1": (4.0, 100.0)},
        )
        project.set_trial(key, per_roi_times=UNSET)
        self.assertIsNone(project.entry(key).trial)
        self.assertEqual(
            project.effective_settings(key).per_roi_times,
            {"mouse_1": (0.0, 360.0)},
        )

    def test_invalid_window_is_reported(self):
        project = self._project()
        key = project.videos[0].key
        project.set_trial(key, per_roi_times={"mouse_1": (50.0, 50.0)})
        self.assertEqual(project.trial_status(key), TRIAL_INVALID)
        self.assertIn("mouse_1", project.trial_problem(key))
        with self.assertRaises(ProjectError):
            project.set_trial(key, per_roi_times={"mouse_1": (1.0,)})

    def test_windows_do_not_touch_start_and_end(self):
        project = self._project()
        key = project.videos[0].key
        project.set_trial(key, per_roi_times={"mouse_1": (10.0, 20.0)})
        composed = project.effective_settings(key)
        self.assertEqual(composed.start_time, 0.0)
        self.assertIsNone(composed.end_time)


class TrialStatusTests(ProjectFixture):
    def _project(self, mode=START_VIDEO):
        project = Project()
        project.set_template(paradigm_settings("Y-Maze"))
        project.add_videos([self.write_video("a.mp4")])
        project.set_trial_protocol(start_mode=mode)
        return project, project.videos[0].key

    def test_ok_when_every_video_starts_with_the_video(self):
        project, key = self._project()
        self.assertEqual(project.trial_status(key), TRIAL_OK)

    def test_per_video_start_needs_a_start(self):
        project, key = self._project(START_PER_VIDEO)
        self.assertEqual(project.trial_status(key), TRIAL_NEEDS_START)
        project.set_trial(key, end=100.0)       # an end is not a start
        self.assertEqual(project.trial_status(key), TRIAL_NEEDS_START)
        project.set_trial(key, start=0.0)       # 0 is a deliberate start
        self.assertEqual(project.trial_status(key), TRIAL_OK)
        project.set_trial(key, start=UNSET)
        self.assertEqual(project.trial_status(key), TRIAL_NEEDS_START)

    def test_legacy_copy_with_a_start_counts_as_having_one(self):
        project, key = self._project(START_PER_VIDEO)
        project.customize(key, paradigm_settings("Y-Maze", offset=2, start_time=8.0))
        self.assertEqual(project.trial_status(key), TRIAL_OK)
        project.customize(key, paradigm_settings("Y-Maze", offset=3))
        self.assertEqual(project.trial_status(key), TRIAL_NEEDS_START)

    def test_invalid_times(self):
        project, key = self._project()
        project.set_trial(key, start=50.0, end=50.0)
        self.assertEqual(project.trial_status(key), TRIAL_INVALID)
        project.set_trial(key, end=40.0)
        self.assertEqual(project.trial_status(key), TRIAL_INVALID)
        project.set_trial(key, end=60.0)
        self.assertEqual(project.trial_status(key), TRIAL_OK)
        project.set_trial(key, start=-1.0)
        self.assertEqual(project.trial_status(key), TRIAL_INVALID)
        # Invalid wins over a missing start.
        project.set_trial_protocol(start_mode=START_PER_VIDEO)
        self.assertEqual(project.trial_status(key), TRIAL_INVALID)

    def test_values_that_are_not_numbers_are_refused(self):
        project, key = self._project()
        for bad in (float("nan"), float("inf"), "5", True):
            with self.assertRaises(ProjectError, msg=repr(bad)):
                project.set_trial(key, start=bad)
            with self.assertRaises(ProjectError, msg=repr(bad)):
                project.set_trial(key, end=bad)
        self.assertIsNone(project.entry(key).trial)

    def test_status_accepts_an_entry_as_well_as_a_key(self):
        project, key = self._project(START_PER_VIDEO)
        entry = project.entry(key)
        self.assertEqual(project.trial_status(entry), TRIAL_NEEDS_START)
        self.assertEqual(project.zone_status(entry), ZONES_TEMPLATE)

    def test_validate_reports_trial_problems_per_video(self):
        project, key = self._project(START_PER_VIDEO)
        codes = {(i.code, i.key) for i in project.validate()}
        self.assertIn(("trial_needs_start", key), codes)
        project.set_trial(key, start=9.0, end=3.0)
        codes = {(i.code, i.key) for i in project.validate()}
        self.assertIn(("trial_invalid", key), codes)
        self.assertNotIn(("trial_needs_start", key), codes)
        project.set_trial(key, end=30.0)
        self.assertEqual(
            [i.code for i in project.validate() if i.key == key
             and i.code.startswith("trial")], []
        )


class ZoneStatusTests(ProjectFixture):
    def _project(self):
        project = Project()
        project.set_template(paradigm_settings("Y-Maze"))
        project.add_videos([self.write_video("a.mp4")])
        return project, project.videos[0].key

    def test_statuses(self):
        project, key = self._project()
        self.assertEqual(project.zone_status(key), ZONES_TEMPLATE)
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        self.assertEqual(project.zone_status(key), ZONES_ADJUSTED)
        project.set_zone_override(
            key, adjusted_zones("Y-Maze"), source=ZONE_ALIGNED
        )
        self.assertEqual(project.zone_status(key), ZONES_ALIGNED)
        self.assertTrue(project.clear_zone_override(key))
        self.assertFalse(project.clear_zone_override(key))
        self.assertEqual(project.zone_status(key), ZONES_TEMPLATE)
        project.customize(key, paradigm_settings("Y-Maze", offset=3))
        self.assertEqual(project.zone_status(key), ZONES_CUSTOM)
        project.customize(
            key, paradigm_settings("Y-Maze", offset=4), source="aligned"
        )
        self.assertEqual(project.zone_status(key), ZONES_ALIGNED)
        project.apply_template_to([key])
        self.assertEqual(project.zone_status(key), ZONES_TEMPLATE)

    def test_a_trial_alone_keeps_the_zones_on_the_template(self):
        project, key = self._project()
        project.set_trial(key, start=5.0)
        self.assertEqual(project.zone_status(key), ZONES_TEMPLATE)

    def test_override_is_copied_and_records_the_template_reference(self):
        project, key = self._project()
        manager = adjusted_zones("Y-Maze")
        zones = project.set_zone_override(key, manager)
        manager.rois.clear()
        self.assertTrue(project.entry(key).zones.roi_manager.rois)
        self.assertEqual(zones.reference, project.zones_reference())
        self.assertIsNotNone(zones.reference)
        self.assertIsNone(zones.transform)

    def test_invalid_arguments(self):
        project, key = self._project()
        with self.assertRaises(ProjectError):
            project.set_zone_override(key, adjusted_zones("Y-Maze"), source="x")
        with self.assertRaises(ProjectError):
            project.set_zone_override(
                key, adjusted_zones("Y-Maze"), transform=[[1, 0], [0, 1]]
            )
        with self.assertRaises(ProjectError):
            project.set_zone_override(
                key, adjusted_zones("Y-Maze"), scale_factor=-1.0
            )
        self.assertIsNone(project.entry(key).zones)

    def test_validate_checks_the_frame_size_of_the_regions(self):
        project, key = self._project()
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        self.assertNotIn(
            "zones_size_mismatch", [i.code for i in project.validate()]
        )
        project.set_video_info(key, frame_size=(1280, 720))
        issue = next(
            i for i in project.validate() if i.code == "zones_size_mismatch"
        )
        self.assertEqual(issue.key, key)
        self.assertIn("1280x720", issue.message)
        self.assertIn("640x480", issue.message)
        project.set_video_info(key, frame_size=(640, 480))
        self.assertNotIn(
            "zones_size_mismatch", [i.code for i in project.validate()]
        )

    def test_validate_notes_when_the_template_regions_changed(self):
        project, key = self._project()
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        self.assertNotIn("zones_outdated", [i.code for i in project.validate()])
        project.set_template(paradigm_settings("Y-Maze", offset=50))
        issue = next(i for i in project.validate() if i.code == "zones_outdated")
        self.assertEqual(issue.key, key)


class MutationTests(ProjectFixture):
    def _project(self):
        project = Project()
        project.set_template(paradigm_settings("Y-Maze"))
        project.add_videos([self.write_video("a.mp4")])
        self.calls = []
        project.add_listener(lambda: self.calls.append(1))
        return project, project.videos[0].key

    def test_every_mutation_notifies(self):
        project, key = self._project()
        actions = [
            lambda: project.set_trial(key, start=3.0),
            lambda: project.clear_trial(key),
            lambda: project.set_trial_protocol(duration_s=60.0),
            lambda: project.apply_protocol_to_all(),
            lambda: project.set_zone_override(key, adjusted_zones("Y-Maze")),
            lambda: project.clear_zone_override(key),
            lambda: project.set_video_info(key, duration_s=90.0),
        ]
        for index, action in enumerate(actions, 1):
            action()
            self.assertEqual(len(self.calls), index, msg=f"action {index}")

    def test_set_trial_merges_and_unsets(self):
        project, key = self._project()
        trial = project.set_trial(key, start=10.0)
        self.assertEqual((trial.start_time, trial.end_time, trial.start_set),
                         (10.0, None, True))
        trial = project.set_trial(key, end=70.0)          # start kept
        self.assertEqual((trial.start_time, trial.end_time), (10.0, 70.0))
        trial = project.set_trial(key, end=UNSET)
        self.assertEqual((trial.start_time, trial.end_time), (10.0, None))
        self.assertIsNone(project.set_trial(key, start=UNSET))
        self.assertIsNone(project.entry(key).trial)

    def test_clear_trial(self):
        project, key = self._project()
        self.assertFalse(project.clear_trial(key))
        project.set_trial(key, start=1.0)
        self.assertTrue(project.clear_trial(key))
        self.assertIsNone(project.entry(key).trial)

    def test_trial_change_marks_results_stale_but_keeps_verification(self):
        project, key = self._project()
        entry = project.entry(key)
        entry.verified = True
        entry.last_status = STATUS_DONE
        project.set_trial(key, start=3.0)
        self.assertTrue(entry.verified)
        self.assertEqual(entry.last_status, STATUS_CONFIGURED)
        entry.last_status = STATUS_DONE
        project.set_trial(key, start=3.0)                # nothing changed
        self.assertEqual(entry.last_status, STATUS_DONE)

    def test_zone_change_drops_verification_and_results(self):
        project, key = self._project()
        entry = project.entry(key)
        entry.verified = True
        entry.last_status = STATUS_DONE
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        self.assertFalse(entry.verified)
        self.assertEqual(entry.last_status, STATUS_CONFIGURED)

    def test_customize_replaces_zones_and_rewrites_the_trial(self):
        project, key = self._project()
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        project.set_trial(key, start=10.0, end=20.0)
        project.customize(
            key, paradigm_settings("Y-Maze", offset=3, start_time=11.0, end_time=22.0)
        )
        entry = project.entry(key)
        self.assertIsNone(entry.zones)
        self.assertEqual((entry.trial.start_time, entry.trial.end_time), (11.0, 22.0))
        self.assertTrue(entry.trial.start_set)
        self.assertEqual(
            project.effective_settings(key).to_dict(),
            paradigm_settings(
                "Y-Maze", offset=3, start_time=11.0, end_time=22.0
            ).to_dict(),
        )

    def test_apply_template_clears_zones_but_keeps_trials(self):
        project, key = self._project()
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        project.set_trial(key, start=10.0)
        project.apply_template_to([key])
        self.assertIsNone(project.entry(key).zones)
        self.assertEqual(project.entry(key).trial.start_time, 10.0)
        project.apply_template_to([key], keep_trials=False)
        self.assertIsNone(project.entry(key).trial)

    def test_set_template_keeps_overrides_on_the_new_template(self):
        project, key = self._project()
        project.set_trial(key, start=10.0, end=50.0)
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        project.set_template(
            paradigm_settings("Y-Maze", offset=60, scale=0.5)
        )
        composed = project.effective_settings(key)
        self.assertEqual(composed.scale_factor, 0.5)          # the template's
        self.assertEqual(composed.start_time, 10.0)           # the trial's
        self.assertEqual(
            composed.roi_manager.to_dict(), adjusted_zones("Y-Maze").to_dict()
        )                                                     # the override's

    def test_changing_the_paradigm_drops_zones_and_windows(self):
        project, key = self._project()
        project.set_zone_override(key, adjusted_zones("Y-Maze"))
        project.set_trial(key, start=10.0, per_roi_times={"m": (1.0, 2.0)})
        project.set_paradigm("Elevated Plus Maze")
        entry = project.entry(key)
        self.assertIsNone(entry.zones)
        self.assertEqual(entry.trial.per_roi_times, {})
        self.assertEqual(entry.trial.start_time, 10.0)


class SmallTypeTests(unittest.TestCase):
    def test_trial_window_round_trips_and_copies(self):
        trial = TrialWindow(
            12, 312.5, {"mouse_1": (1.0, 2.0)}, start_set=True
        )
        self.assertEqual(TrialWindow.from_dict(trial.to_dict()), trial)
        clone = trial.copy()
        clone.per_roi_times["mouse_2"] = (3.0, 4.0)
        self.assertNotIn("mouse_2", trial.per_roi_times)
        self.assertFalse(TrialWindow().start_set)
        self.assertTrue(TrialWindow().is_empty())
        self.assertFalse(trial.is_empty())

    def test_trial_window_rejects_garbage(self):
        for bad in (
            {"start_time": "x"}, {"end_time": float("nan")},
            {"per_roi_times": {"m": [1]}}, {"per_roi_times": [1]}, [],
        ):
            with self.assertRaises(ProjectError, msg=repr(bad)):
                TrialWindow.from_dict(bad)

    def test_zone_override_round_trips(self):
        zones = ZoneOverride(
            paradigm_settings("Y-Maze", offset=5).roi_manager, ZONE_ALIGNED,
            [[1, 0, 2], [0, 1, 3], [0, 0, 1]], "abc", 0.25,
        )
        loaded = ZoneOverride.from_dict(zones.to_dict())
        self.assertEqual(loaded.to_dict(), zones.to_dict())
        self.assertEqual(loaded.source, ZONE_ALIGNED)
        with self.assertRaises(ProjectError):
            ZoneOverride.from_dict({"source": "adjusted"})
        with self.assertRaises(ProjectError):
            ZoneOverride.from_dict(
                {"roi_manager": zones.roi_manager.to_dict(), "source": "weird"}
            )

    def test_protocol_defaults(self):
        protocol = TrialProtocol()
        self.assertIsNone(protocol.duration_s)
        self.assertEqual(protocol.start_mode, START_VIDEO)
        self.assertEqual(TrialProtocol.from_dict(None), protocol)
        self.assertEqual(
            TrialProtocol.from_dict(TrialProtocol(60.0, START_PER_VIDEO).to_dict()),
            TrialProtocol(60.0, START_PER_VIDEO),
        )



if __name__ == "__main__":
    unittest.main()
