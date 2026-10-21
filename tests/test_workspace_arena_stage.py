"""The Arena & zones stage: live template editing, per-video zones, alignment.

The stage must write exactly what the old Setup tab (and the classic wizard)
write for the same drawing, apply every edit live (debounced, flushed on
leave), keep per-video zones in ``Project.zones`` and align videos through
``core.roi_alignment``.
"""
import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core import roi_alignment as ra
from core.models import AppConfig, BatchSettings
from core.project import (ZONE_ADJUSTED, ZONE_ALIGNED, ZONES_ADJUSTED,
                          ZONES_ALIGNED, ZONES_CUSTOM, ZONES_TEMPLATE, Project)
from core.setup_state import SetupState
from core.video import ReferenceFrameGenerator
from mazes import create_maze
from roi_manager import ROI, ROIManager
from tests.alignment_support import arena_image, camera_move, max_grid_error, warp
from tests.project_support import ProjectFixture
from tests.setup_tab_support import (E, F, T, Y, SetupTabFixture, draw_polygon,
                                     draw_zones, wait_until)
from tests.synthetic_video import write_indexed_video
from tests.test_workspace_setup_equivalence import (ARENA, LINE,
                                                    EquivalenceFixture,
                                                    arm_zones)
from tests.test_workspace_setup_tab import valid_settings
from tests.test_workspace_stage_status import settings_for
from tests.workspace_support import HAS_RUNTIME

if HAS_RUNTIME:
    from PyQt6.QtCore import Qt
    from gui.background_task import TaskCancelled
    from gui.roi_editor import TOOL_LINE, RoiEditorWidget
    from gui.setup_panels import ParadigmParamsPanel
    from gui.workspace.stages import arena_stage as arena
    from gui.workspace.stages.arena_stage import (ArenaStage, MODE_TEMPLATE,
                                                  MODE_VIDEO, PAGE_EDITOR,
                                                  PAGE_MESSAGE, PAGE_READONLY)
    from gui.workspace.stages.hosted import HostedStage
    from gui.workspace.ui_kit import CollapsibleSection

BIG = (200, 150)
RUN = "gui.workspace.stages.arena_stage.run_with_progress"


def scaling(factor=1.25, verdict=ra.VERDICT_GOOD) -> "ra.RegistrationResult":
    return ra.RegistrationResult(
        np.diag([factor, factor, 1.0]), 50, 60, 0.4, verdict, "")


def failed_registration() -> "ra.RegistrationResult":
    return ra.RegistrationResult(None, 0, 3, 0.0, ra.VERDICT_FAILED,
                                 "not enough of the arena matched")


# ----------------------------------------------------------------------
# The Qt-free rules
# ----------------------------------------------------------------------
class RuleTests(ProjectFixture):
    def make(self, template=True, videos=3):
        project = Project(name="Study", paradigm=Y)
        if template:
            settings = settings_for(Y, size=(160, 120))
            settings.roi_manager.set_reference_line((10, 100), (110, 100), "User-defined")
            project.set_template(settings)
        project.add_videos([
            self.write_video(f"v{n}.mp4", b"x" * (n + 1)) for n in range(videos)
        ])
        for entry in project.videos:
            project.set_video_info(entry.key, frame_size=(160, 120))
        return project

    def badge(self, project, index, codes=()):
        return arena.zone_badge(project, project.videos[index], set(codes))

    def test_a_video_that_follows_the_template(self):
        project = self.make()
        self.assertEqual(self.badge(project, 0), ("Template", "neutral"))

    def test_adjusted_zones(self):
        project = self.make()
        manager = project.template_settings.roi_manager
        project.set_zone_override(project.videos[0].key, manager, ZONE_ADJUSTED)
        self.assertEqual(self.badge(project, 0), ("Adjusted", "info"))

    def test_aligned_zones(self):
        project = self.make()
        manager = project.template_settings.roi_manager
        project.set_zone_override(project.videos[0].key, manager, ZONE_ALIGNED)
        self.assertEqual(self.badge(project, 0), ("Aligned", "ok"))

    def test_a_legacy_copy_is_custom(self):
        project = self.make()
        project.customize(project.videos[0].key, project.template_settings.copy())
        self.assertEqual(project.zone_status(project.videos[0]), ZONES_CUSTOM)
        self.assertEqual(self.badge(project, 0), ("Custom", "neutral"))

    def test_another_frame_size_needs_a_check(self):
        project = self.make()
        project.set_video_info(project.videos[1].key, frame_size=(320, 240))
        self.assertEqual(self.badge(project, 1), ("Check alignment", "warning"))
        self.assertEqual(self.badge(project, 0), ("Template", "neutral"))

    def test_the_validate_codes_need_a_check(self):
        project = self.make()
        for code in ("zones_size_mismatch", "zones_outdated"):
            with self.subTest(code):
                self.assertEqual(
                    self.badge(project, 0, [code]), ("Check alignment", "warning"))

    def test_the_check_wins_over_the_zone_source(self):
        project = self.make()
        manager = project.template_settings.roi_manager
        project.set_zone_override(project.videos[0].key, manager, ZONE_ADJUSTED)
        self.assertEqual(
            self.badge(project, 0, ["zones_outdated"])[0], "Check alignment")

    def test_an_adjusted_video_of_another_size_is_fine(self):
        project = self.make()
        key = project.videos[0].key
        project.set_video_info(key, frame_size=(320, 240))
        manager = ROIManager(source_dimensions=(320, 240))
        manager.add_roi("arm_a", np.array([[0, 0], [60, 0], [60, 60], [0, 60]]))
        project.set_zone_override(key, manager, ZONE_ADJUSTED)
        self.assertEqual(self.badge(project, 0), ("Adjusted", "info"))

    def test_the_counts(self):
        project = self.make(videos=4)
        manager = project.template_settings.roi_manager
        project.set_zone_override(project.videos[0].key, manager, ZONE_ADJUSTED)
        project.set_video_info(project.videos[1].key, frame_size=(320, 240))
        counts = arena.zone_counts(project, project.validate())
        self.assertEqual((counts.template, counts.own, counts.check), (3, 1, 1))

    def test_no_template_every_video_is_plain(self):
        project = self.make(template=False)
        self.assertEqual(self.badge(project, 0), ("Template", "neutral"))
        self.assertIsNone(arena.template_dimensions(project))

    def test_the_scale_of_a_video_that_keeps_the_template_line(self):
        project = self.make()
        state = arena.template_state(project)
        state.scale_cm = 25.0
        project.set_template(state.to_batch_settings(finalize=True))
        template = project.template_settings
        self.assertGreater(template.scale_factor, 0)
        self.assertIsNone(
            arena.scale_factor_for(project, copy.deepcopy(template.roi_manager)))

    def test_the_scale_of_a_video_with_a_longer_line(self):
        project = self.make()
        state = arena.template_state(project)
        state.scale_cm = 25.0
        project.set_template(state.to_batch_settings(finalize=True))
        manager = copy.deepcopy(project.template_settings.roi_manager)
        manager.set_reference_line((0, 100), (200, 100), "User-defined")
        self.assertAlmostEqual(arena.scale_factor_for(project, manager), 25.0 / 200)

    def test_a_video_without_a_line_keeps_the_template_scale(self):
        project = self.make()
        manager = ROIManager(source_dimensions=(160, 120))
        manager.add_roi("arm_a", np.array([[0, 0], [60, 0], [60, 60], [0, 60]]))
        self.assertIsNone(arena.scale_factor_for(project, manager))


# ----------------------------------------------------------------------
# The stage on a real window
# ----------------------------------------------------------------------
class ArenaFixture(SetupTabFixture):
    """Two real 160x120 clips, a paradigm and (by default) a valid template."""

    PARADIGM = Y
    TEMPLATE = True

    def setUp(self):
        super().setUp()
        cache = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(cache.cleanup)
        patcher = patch.object(
            ReferenceFrameGenerator, "get_cache_dir", return_value=Path(cache.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        if self.TEMPLATE and self.PARADIGM:
            self.project.set_template(valid_settings(self.PARADIGM))
            self.app.processEvents()
        self.stage = self.window.stage("arena")

    # -- helpers ------------------------------------------------------------
    def settle(self):
        self.window.thumbnails.wait_idle()
        for _ in range(3):
            self.app.processEvents()

    def enter(self):
        self.assertTrue(self.window.go_to_stage("arena"))
        self.settle()
        return self.stage

    def add_big(self, count=2):
        folder = self.root / "big"
        folder.mkdir(exist_ok=True)
        clips = [write_indexed_video(folder / f"big{n}.avi", 40, size=BIG)
                 for n in range(count)]
        self.project.add_videos(clips)
        self.app.processEvents()
        return self.project.keys()[2:]

    def template_zones(self):
        return self.project.template_settings.roi_manager

    def effective_zones(self, key):
        return self.project.effective_settings(key).roi_manager


class StageTests(ArenaFixture):
    def test_it_is_a_real_stage_not_a_hosted_one(self):
        self.assertIsInstance(self.stage, ArenaStage)
        self.assertNotIsInstance(self.stage, HostedStage)
        self.assertEqual(self.stage.stage_id, "arena")
        self.assertEqual(self.stage.header.title(), "Arena & zones")

    def test_entering_shows_the_template_in_the_editor(self):
        stage = self.enter()
        self.assertEqual(stage.mode, MODE_TEMPLATE)
        self.assertEqual(stage.stack.currentIndex(), PAGE_EDITOR)
        self.assertIsInstance(stage.editor, RoiEditorWidget)
        self.assertEqual(stage.editor.roi_manager().to_dict(),
                         self.template_zones().to_dict())

    def test_without_a_project_there_is_a_message(self):
        self.prompts.say("ask_save_changes", "discard")
        self.window.close_project()
        self.window.go_to_stage("arena", force=True)
        self.assertEqual(self.stage.stack.currentIndex(), PAGE_MESSAGE)
        self.assertIsNone(self.stage.editor)

    def test_the_parts_of_the_page(self):
        stage = self.enter()
        self.assertEqual(stage.editing_card.title(), "Editing")
        self.assertEqual(stage.calibration_card.title(), "Calibration")
        self.assertIsInstance(stage.params_section, CollapsibleSection)
        self.assertEqual(stage.params_section.title(), "Paradigm options")
        self.assertIsInstance(stage.params, ParadigmParamsPanel)
        self.assertEqual(stage.radio_template.text(), "All videos (template)")
        self.assertEqual(stage.radio_video.text(), "This video only")
        self.assertTrue(stage.radio_template.isChecked())
        self.assertEqual({h.text() for h in stage.key_hints},
                         {"D", "E", "L", "Ctrl+Z", "F"})

    def test_the_footer_names_the_next_stage(self):
        self.assertEqual(self.stage.next_label(), "Detection")

    def test_tst_leads_to_motion_tuning(self):
        self.prompts.say("confirm", True)
        self.choose_paradigm(T)
        self.assertEqual(self.stage.next_label(), "Motion tuning")

    def test_the_video_list_shows_one_badge_per_video(self):
        stage = self.enter()
        items = stage.nav.items()
        self.assertEqual([i.key for i in items], self.keys)
        self.assertEqual({i.badge_text for i in items}, {"Template"})
        self.assertEqual({b.kind for b in stage.nav.rail.buttons()}, {"neutral"})

    def test_the_picked_video_is_the_reference(self):
        stage = self.enter()
        self.select([self.keys[1]])
        self.window.go_to_stage("videos")
        self.window.go_to_stage("arena")
        self.assertEqual(stage.selected_key, self.keys[1])
        self.assertEqual(stage.nav.selected_key(), self.keys[1])

    def test_picking_a_video_in_template_mode_changes_the_frame_only(self):
        stage = self.enter()
        before = stage.editor.roi_manager().to_dict()
        self.assertTrue(stage.pick_video(self.keys[1]))
        self.assertEqual(stage.mode, MODE_TEMPLATE)
        self.assertEqual(stage.selected_key, self.keys[1])
        self.assertEqual(stage.editor.roi_manager().to_dict(), before)
        self.assertFalse(stage.pick_video("no such video"))

    def test_the_calibration_button_switches_to_the_line_tool(self):
        stage = self.enter()
        self.assertTrue(stage.draw_line_button.isEnabled())
        stage.draw_line_button.click()
        self.assertEqual(stage.editor.tool(), TOOL_LINE)

    def test_the_calibration_card_is_hidden_when_the_test_has_no_distances(self):
        self.prompts.say("confirm", True, True)
        self.choose_paradigm(T)
        self.project.set_template(valid_settings(T))
        stage = self.enter()
        self.assertTrue(stage.calibration_card.isHidden())
        self.choose_paradigm(Y)
        self.project.set_template(valid_settings(Y))
        self.window.go_to_stage("videos")
        stage = self.enter()
        self.assertFalse(stage.calibration_card.isHidden())


class PaintTests(ArenaFixture):
    """Every page of the stage paints (an exception in a paint event would
    abort the process outside the tests)."""

    def paint(self):
        self.window.resize(1366, 768)
        self.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        self.window.show()
        self.app.processEvents()
        self.assertFalse(self.window.grab().isNull())

    def test_every_page_paints(self):
        big = self.add_big(1)
        stage = self.enter()
        self.paint()                                    # editor, rail
        stage.nav.set_collapsed(False)
        self.paint()                                    # editor, filmstrip
        stage.set_mode(MODE_VIDEO)
        stage.pick_video(self.keys[1])
        self.paint()                                    # read-only zones
        self.assertFalse(stage.readonly.grab().isNull())
        stage.pick_video(big[0])
        self.paint()                                    # misplaced zones
        stage.adjust_video()
        self.paint()                                    # video editor
        self.project.set_template(None)
        self.app.processEvents()
        self.paint()                                    # "draw the template first"

    def test_the_read_only_view_draws_the_zones_over_the_frame(self):
        stage = self.enter()
        stage.set_mode(MODE_VIDEO)
        stage.pick_video(self.keys[0])
        view = stage.readonly
        view.resize(480, 360)
        image = view.grab().toImage()
        colors = {image.pixelColor(x, y).name()
                  for x in range(0, 480, 6) for y in range(0, 360, 6)}
        self.assertGreater(len(colors), 3)              # frame + zone tints


class WidthTests(ArenaFixture):
    def shown(self, width, height):
        self.window.resize(width, height)
        self.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        self.window.show()
        self.app.processEvents()
        stage = self.enter()
        self.app.processEvents()
        return stage

    def test_the_editor_takes_over_65_percent_at_1366x768(self):
        stage = self.shown(1366, 768)
        self.assertTrue(stage.nav.is_collapsed())
        self.assertGreaterEqual(
            stage.editor.width(), 0.65 * self.window.width(), stage.editor.width())

    def test_and_at_1920x1080_with_the_full_list(self):
        stage = self.shown(1920, 1080)
        self.assertFalse(stage.nav.is_collapsed())
        self.assertGreaterEqual(
            stage.editor.width(), 0.65 * self.window.width(), stage.editor.width())

    def test_the_users_choice_of_list_is_remembered(self):
        stage = self.shown(1366, 768)
        stage.nav.toggle()                      # open it by hand
        self.assertFalse(stage.nav.is_collapsed())
        self.assertFalse(self.settings.ui_flag(arena.STRIP_FLAG, default=True))
        # A fresh window of the same settings starts as the user left it.
        from gui.workspace.window import WorkspaceWindow
        other = WorkspaceWindow(self.settings, self.prompts)
        self.addCleanup(other.deleteLater)
        other.resize(1366, 768)
        other._install_project(self.window.project)
        other.stage("arena").on_enter()
        self.assertFalse(other.stage("arena").nav.is_collapsed())
        other._detach_project()

    def test_the_rail_and_the_strip_pick_the_same_video(self):
        stage = self.shown(1366, 768)
        picked = []
        stage.nav.video_selected.connect(picked.append)
        stage.nav.rail.buttons()[1].click()
        self.assertEqual(picked, [self.keys[1]])
        self.assertEqual(stage.selected_key, self.keys[1])
        self.assertEqual(stage.nav.strip.selected_key(), self.keys[1])


class VideoFactsTests(ArenaFixture):
    def test_entering_records_the_frame_size_and_duration(self):
        for key in self.keys:
            self.assertIsNone(self.project.entry(key).frame_size)
        self.enter()
        for key in self.keys:
            entry = self.project.entry(key)
            self.assertEqual(entry.frame_size, (160, 120))
            self.assertGreater(entry.duration_s, 0)

    def test_a_bigger_video_is_flagged_once_its_size_is_known(self):
        big_keys = self.add_big(1)
        stage = self.enter()
        item = {i.key: i for i in stage.nav.items()}[big_keys[0]]
        self.assertEqual(item.badge_text, "Check alignment")
        self.assertEqual(item.badge_kind, "warning")
        codes = {i.code for i in self.project.validate() if i.key == big_keys[0]}
        self.assertIn("zones_size_mismatch", codes)

    def test_nothing_is_rewritten_the_second_time(self):
        self.enter()
        calls = []
        self.project.add_listener(lambda: calls.append(1))
        self.window.go_to_stage("videos")
        self.enter()
        self.assertEqual(calls, [])


class TemplateLiveTests(ArenaFixture):
    TEMPLATE = False

    def zones(self):
        return arm_zones(create_maze(Y))

    def wrap_set_template(self):
        calls = []
        original = self.project.set_template

        def spy(settings, *args, **kwargs):
            calls.append(settings)
            return original(settings, *args, **kwargs)
        patcher = patch.object(self.project, "set_template", side_effect=spy)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_drawing_is_written_after_a_pause_in_one_go(self):
        stage = self.enter()
        calls = self.wrap_set_template()
        draw_zones(stage.editor, self.zones())
        self.assertIsNone(self.project.template_settings)
        self.assertTrue(stage.has_pending_writes())
        self.assertTrue(wait_until(self.app, lambda: bool(calls), 5.0))
        self.assertEqual(len(calls), 1)
        self.assertFalse(stage.has_pending_writes())
        self.assertEqual(
            set(self.template_zones().rois), set(self.zones()))

    def test_edits_inside_the_pause_are_one_write(self):
        stage = self.enter()
        calls = self.wrap_set_template()
        for index, name in enumerate(self.zones()):
            draw_zones(stage.editor, [name], first_index=index)
        stage.flush()
        self.assertEqual(len(calls), 1)

    def test_leaving_the_stage_writes_what_is_pending(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones())
        self.assertIsNone(self.project.template_settings)
        self.assertTrue(self.window.go_to_stage("videos"))
        self.assertIsNotNone(self.project.template_settings)
        self.assertEqual(self.prompts.called("ask_save_changes"), [])

    def test_asking_whether_next_is_allowed_writes_first(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones())
        self.assertTrue(stage.has_pending_writes())
        self.assertEqual(stage.can_advance(), (True, ""))
        self.assertIsNotNone(self.project.template_settings)

    def test_closing_the_project_writes_what_is_pending(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones())
        self.prompts.say("ask_save_changes", "discard")
        self.assertTrue(self.window.close_project())
        self.assertFalse(stage.has_pending_writes())
        self.assertEqual(len(self.prompts.called("ask_save_changes")), 1)

    def test_followers_receive_the_template(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones())
        stage.flush()
        template = self.project.template_settings
        for key in self.keys:
            self.assertEqual(
                self.project.entry(key).settings.to_dict(), template.to_dict())

    def test_an_incomplete_drawing_is_kept_and_blocks_next(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones()[:2])
        stage.flush()
        self.assertIsNotNone(self.project.template_settings)
        ok, reason = stage.can_advance()
        self.assertFalse(ok)
        self.assertTrue(reason)
        self.assertEqual(stage.status()[0], "attention")

    def test_before_anything_is_drawn_there_is_nothing_to_write(self):
        stage = self.enter()
        stage.flush()
        self.assertIsNone(self.project.template_settings)
        self.assertEqual(stage.status()[0], "todo")
        ok, reason = stage.can_advance()
        self.assertFalse(ok)
        self.assertIn("Draw", reason)
        self.assertEqual(stage.current_notice()[0], "tip")

    def test_the_calibration_length_is_written(self):
        stage = self.enter()

        def draw(editor):
            draw_zones(editor, self.zones())
            self.assertTrue(editor.set_line(*LINE))
        draw(stage.editor)
        stage.flush()
        self.assertEqual(self.project.template_settings.scale_factor, 0.0)
        stage.calibration.scale_spin.setValue(25.0)
        stage.flush()
        self.assertAlmostEqual(
            self.project.template_settings.scale_factor, 25.0 / 100.0)
        # Redrawing the line keeps the length and follows the new pixels.
        self.assertTrue(stage.editor.set_line((10, 100), (150, 100)))
        stage.flush()
        self.assertAlmostEqual(
            self.project.template_settings.scale_factor, 25.0 / 140.0)

    def test_paradigm_options_are_written_and_rebuild_the_editor(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones())
        stage.flush()
        panel = stage.params.active_panel
        panel.dwell_spin.setValue(0.4)
        panel.initial_check.setChecked(False)
        stage.flush()
        parameters = self.project.template_settings.algorithm_parameters
        self.assertAlmostEqual(parameters["min_dwell_seconds"], 0.4)
        self.assertFalse(parameters["count_initial_arm"])
        # The zones stayed.
        self.assertEqual(set(self.template_zones().rois), set(self.zones()))

    def test_the_tst_mouse_count_rebuilds_the_editor_rules(self):
        self.choose_paradigm(T)
        stage = self.enter()
        stage.params.active_panel.mice_spin.setValue(2)
        stage.flush()
        rows = [row.category for row in stage.editor.session.rows()]
        self.assertEqual(rows, ["mouse_1", "mouse_2"])
        draw_zones(stage.editor, ["mouse_1", "mouse_2"])
        stage.flush()
        self.assertEqual(
            self.project.template_settings.algorithm_parameters["num_mice"], 2)

    def test_options_set_before_any_zone_are_kept_for_the_first_write(self):
        stage = self.enter()
        stage.params.active_panel.dwell_spin.setValue(0.35)
        stage.flush()
        self.assertIsNone(self.project.template_settings)
        draw_zones(stage.editor, self.zones())
        stage.flush()
        self.assertAlmostEqual(
            self.project.template_settings.algorithm_parameters["min_dwell_seconds"],
            0.35)

    def test_an_edit_made_while_a_run_is_locked_is_not_written(self):
        self.enter()
        self.window.set_busy(True)
        self.addCleanup(self.window.set_busy, False)
        self.window.go_to_stage("videos")
        stage = self.enter()
        self.assertFalse(stage.side.isEnabled())
        self.assertFalse(stage.editor.isEnabled())
        self.assertEqual(stage.current_notice()[0], "info")
        draw_zones(stage.editor, self.zones())
        stage.flush()
        self.assertIsNone(self.project.template_settings)

    def test_the_other_stages_edits_of_the_template_are_picked_up(self):
        stage = self.enter()
        draw_zones(stage.editor, self.zones())
        stage.flush()
        edited = self.project.template_settings.copy()
        edited.detection_config.min_component_area = 77
        self.project.set_template(edited)
        self.app.processEvents()
        stage.flush()
        self.assertEqual(
            self.project.template_settings.detection_config.min_component_area, 77)


# ----------------------------------------------------------------------
# Equivalence with the classic wizard / the Setup tab
# ----------------------------------------------------------------------
class StageEquivalence(EquivalenceFixture):
    """The template this stage stores equals the classic result."""

    def setUp(self):
        super().setUp()
        self.stage = self.window.stage("arena")

    def stage_params(self, edit):
        stage = self.stage
        self.assertTrue(self.window.go_to_stage("arena"))
        edit(stage.params.active_panel)
        stage.flush()

    def stage_rois(self, draw):
        self.assertTrue(self.window.go_to_stage("arena"))
        draw(self.stage.editor)
        self.stage.flush()

    def stage_cm(self, cm):
        self.assertTrue(self.window.go_to_stage("arena"))
        self.stage.calibration.scale_spin.setValue(cm)
        self.stage.flush()

    def check_arms(self, paradigm):
        self.choose_paradigm(paradigm)
        zones = arm_zones(create_maze(paradigm))
        self.stage_params(lambda panel: (
            panel.dwell_spin.setValue(0.3), panel.initial_check.setChecked(False)))

        def draw(editor):
            draw_zones(editor, zones)
            self.assertTrue(editor.set_line(*LINE))
        self.stage_rois(draw)
        self.stage_cm(25.0)

        from gui.arm_entry_dialog import ArmEntryParametersDialog
        maze = create_maze(paradigm)
        roi_manager = self.classic_rois(maze, draw)
        arm = ArmEntryParametersDialog(maze)
        self.addCleanup(arm.deleteLater)
        arm.dwell_spin.setValue(0.3)
        arm.initial_check.setChecked(False)
        maze.configure_algorithm_parameters(arm.get_parameters())
        base = BatchSettings(roi_manager, AppConfig().detection, 0.0)
        classic = self.classic_review(
            base, maze, lambda dialog: dialog.scale_spin.setValue(25.0))
        self.assert_same(classic)
        self.assertGreater(self.project.template_settings.scale_factor, 0)

    def test_y_maze(self):
        self.check_arms(Y)

    def test_elevated_plus_maze(self):
        self.check_arms(E)

    def test_open_field_with_an_arena(self):
        self.choose_paradigm(F)

        def params(panel):
            panel.num_zones_spin.setValue(2)
            panel.quick_fill("letters")
            panel.arena_checkbox.setChecked(True)
            panel.immobility_threshold_spin.setValue(3.5)
        self.stage_params(params)

        def draw(editor):
            draw_polygon(editor, "arena", ARENA)
            draw_zones(editor, ["zone_a", "zone_b"], first_index=1)
            self.assertTrue(editor.set_line(*LINE))
        self.stage_rois(draw)
        self.stage_cm(30.0)

        from gui.freestyle_config_dialog import FreestyleConfigDialog
        maze = create_maze(F)
        dialog = FreestyleConfigDialog(maze=maze)
        self.addCleanup(dialog.deleteLater)
        params(dialog)
        dialog.validate_and_accept()
        maze.configure_zones(dialog.get_zone_definitions())
        roi_manager = self.classic_rois(maze, draw)
        base = BatchSettings(roi_manager, AppConfig().detection, 0.0)
        classic = self.classic_review(
            base, maze, lambda settings: settings.scale_spin.setValue(30.0))
        self.assert_same(classic)
        parameters = self.project.template_settings.algorithm_parameters
        self.assertTrue(parameters["arena_mode"])
        self.assertIsNotNone(self.project.template_settings.roi_manager.arena)

    def test_zone_free_open_field(self):
        self.choose_paradigm(F)
        self.stage_params(lambda panel: panel.zone_free_radio.setChecked(True))

        def draw(editor):
            self.assertTrue(editor.set_line(*LINE))
        self.stage_rois(draw)
        self.stage_cm(12.5)

        from gui.freestyle_config_dialog import FreestyleConfigDialog
        maze = create_maze(F)
        dialog = FreestyleConfigDialog(maze=maze)
        self.addCleanup(dialog.deleteLater)
        dialog.zone_free_radio.setChecked(True)
        dialog.validate_and_accept()
        maze.configure_zones(dialog.get_zone_definitions())
        roi_manager = self.classic_rois(maze, draw)
        base = BatchSettings(roi_manager, AppConfig().detection, 0.0)
        classic = self.classic_review(
            base, maze, lambda settings: settings.scale_spin.setValue(12.5))
        self.assert_same(classic)

    def test_tail_suspension_with_two_mice(self):
        self.choose_paradigm(T)
        self.stage_params(lambda panel: panel.mice_spin.setValue(2))
        self.stage_rois(lambda editor: draw_zones(editor, ["mouse_1", "mouse_2"]))

        maze = create_maze(T)
        maze.configure_mice(2)
        roi_manager = self.classic_rois(
            maze, lambda editor: draw_zones(editor, ["mouse_1", "mouse_2"]))
        state = SetupState.new(T)
        state.set_maze(maze)
        state.roi_manager = roi_manager
        self.assert_same(state.to_batch_settings(finalize=True))
        self.assertEqual(self.project.template_settings.scale_factor, 0.0)

    def test_it_equals_what_the_setup_tab_stores(self):
        """The old tab, driven through its own steps, stores the same."""
        self.choose_paradigm(Y)
        zones = arm_zones(create_maze(Y))

        def draw(editor):
            draw_zones(editor, zones)
            self.assertTrue(editor.set_line(*LINE))
        self.stage_rois(draw)
        self.stage_cm(25.0)
        stored = self.project.template_settings.to_dict()
        self.project.set_template(None)
        self.tab_rois(draw)
        self.tab_calibration(25.0)
        self.assertEqual(self.project.template_settings.to_dict(), stored)


# ----------------------------------------------------------------------
# Zones of a single video
# ----------------------------------------------------------------------
class VideoZonesTests(ArenaFixture):
    def setUp(self):
        super().setUp()
        self.stage = self.enter()
        self.key = self.keys[1]

    def view(self, key=None):
        self.assertTrue(self.stage.set_mode(MODE_VIDEO))
        self.assertTrue(self.stage.pick_video(key or self.key))

    def test_a_following_video_shows_the_template_read_only(self):
        self.view()
        stage = self.stage
        self.assertEqual(stage.stack.currentIndex(), PAGE_READONLY)
        self.assertIsNone(stage.editor)
        self.assertEqual(stage.readonly.manager.to_dict(),
                         self.template_zones().to_dict())
        self.assertEqual(stage.readonly.frame_size, (160, 120))
        self.assertEqual(stage.current_notice(), ("tip", arena.TEMPLATE_NOTE))
        self.assertEqual(
            arena.TEMPLATE_NOTE,
            "These are the template zones. Looks off? Adjust zones for this video.")
        self.assertFalse(stage.video_block.isHidden())
        self.assertTrue(stage.adjust_button.isEnabled())
        self.assertFalse(stage.reset_button.isEnabled())
        self.assertTrue(stage.reset_button.toolTip())

    def test_viewing_changes_nothing_in_the_project(self):
        before = self.project.to_dict()
        self.view()
        self.stage.flush()
        self.assertEqual(self.project.to_dict(), before)

    def test_adjusting_gives_the_video_its_own_zones_at_once(self):
        self.view()
        self.assertTrue(self.stage.adjust_video())
        entry = self.project.entry(self.key)
        self.assertEqual(entry.zones.source, ZONE_ADJUSTED)
        self.assertEqual(self.project.zone_status(entry), ZONES_ADJUSTED)
        self.assertEqual(self.effective_zones(self.key).to_dict(),
                         self.template_zones().to_dict())
        self.assertEqual(self.stage.stack.currentIndex(), PAGE_EDITOR)
        self.assertEqual(self.stage.editor.source_size, (160, 120))
        self.assertFalse(self.stage.adjust_button.isEnabled())
        self.assertTrue(self.stage.reset_button.isEnabled())
        badges = {i.key: i.badge_text for i in self.stage.nav.items()}
        self.assertEqual(badges[self.key], "Adjusted")
        self.assertEqual(badges[self.keys[0]], "Template")

    def test_edits_are_written_live_and_compose_to_the_expected_zones(self):
        self.view()
        self.stage.adjust_video()
        editor = self.stage.editor
        new_arm_a = [(2, 60), (50, 60), (50, 110), (2, 110)]
        editor.set_current_zone("arm_a")
        editor.clear_current_zone()
        draw_polygon(editor, "arm_a", new_arm_a)
        self.assertTrue(self.stage.has_pending_writes())
        self.stage.flush()

        expected = copy.deepcopy(self.template_zones())
        expected.rois["arm_a"] = [ROI(np.array(new_arm_a), "arm_a")]
        self.assertEqual(self.effective_zones(self.key).to_dict(), expected.to_dict())
        self.assertEqual(self.project.entry(self.key).zones.roi_manager.to_dict(),
                         expected.to_dict())
        # Everything else is untouched, the template included.
        self.assertEqual(self.effective_zones(self.keys[0]).to_dict(),
                         self.template_zones().to_dict())
        self.assertEqual(self.template_zones().rois["arm_a"][0].points.tolist(),
                         valid_settings(Y).roi_manager.rois["arm_a"][0].points.tolist())

    def test_the_edit_is_written_after_a_pause_without_asking(self):
        self.view()
        self.stage.adjust_video()
        editor = self.stage.editor
        editor.set_current_zone("arm_a")
        editor.clear_current_zone()
        draw_polygon(editor, "arm_a", [(2, 60), (50, 60), (50, 110), (2, 110)])
        entry = self.project.entry(self.key)
        before = entry.zones.roi_manager.to_dict()
        self.assertTrue(wait_until(
            self.app, lambda: entry.zones.roi_manager.to_dict() != before, 5.0))
        self.assertEqual(self.prompts.called("ask_save_changes"), [])

    def test_leaving_flushes_a_pending_video_edit(self):
        self.view()
        self.stage.adjust_video()
        editor = self.stage.editor
        editor.set_current_zone("arm_a")
        editor.clear_current_zone()
        draw_polygon(editor, "arm_a", [(2, 60), (50, 60), (50, 110), (2, 110)])
        self.assertTrue(self.window.go_to_stage("trials"))
        points = self.effective_zones(self.key).rois["arm_a"][0].points
        self.assertEqual(points.tolist(), [[2, 60], [50, 60], [50, 110], [2, 110]])

    def test_reset_goes_back_to_the_template(self):
        self.view()
        self.stage.adjust_video()
        self.assertTrue(self.stage.reset_video())
        self.assertIsNone(self.project.entry(self.key).zones)
        self.assertEqual(self.project.zone_status(self.key), ZONES_TEMPLATE)
        self.assertEqual(self.stage.stack.currentIndex(), PAGE_READONLY)
        self.assertFalse(self.stage.reset_video())

    def test_the_reset_button(self):
        self.view()
        self.stage.adjust_video()
        self.stage.reset_button.click()
        self.assertEqual(self.project.zone_status(self.key), ZONES_TEMPLATE)

    def test_a_template_edit_does_not_touch_adjusted_zones(self):
        self.view()
        self.stage.adjust_video()
        adjusted = self.project.entry(self.key).zones.roi_manager.to_dict()
        self.stage.set_mode(MODE_TEMPLATE)
        self.stage.editor.set_current_zone("arm_a")
        self.stage.editor.clear_current_zone()
        draw_polygon(self.stage.editor, "arm_a",
                     [(2, 60), (50, 60), (50, 110), (2, 110)])
        self.stage.flush()
        self.assertEqual(
            self.project.entry(self.key).zones.roi_manager.to_dict(), adjusted)
        codes = {i.code for i in self.project.validate() if i.key == self.key}
        self.assertIn("zones_outdated", codes)
        badges = {i.key: i.badge_text for i in self.stage.nav.items()}
        self.assertEqual(badges[self.key], "Check alignment")

    def test_the_video_mode_before_a_template_asks_for_one(self):
        self.project.set_template(None)
        self.app.processEvents()
        self.window.go_to_stage("videos")
        stage = self.enter()
        stage.set_mode(MODE_VIDEO)
        self.assertEqual(stage.stack.currentIndex(), PAGE_MESSAGE)
        self.assertEqual(stage.message.title(), "Draw the template zones first")
        stage.message.buttons[0].click()
        self.assertEqual(stage.mode, MODE_TEMPLATE)
        self.assertEqual(stage.stack.currentIndex(), PAGE_EDITOR)

    def test_radio_buttons_switch_the_mode(self):
        self.stage.radio_video.click()
        self.assertEqual(self.stage.mode, MODE_VIDEO)
        self.stage.radio_template.click()
        self.assertEqual(self.stage.mode, MODE_TEMPLATE)

    def test_a_calibration_change_rescales_the_videos_with_their_own_line(self):
        stage = self.stage
        stage.draw_reference_line()
        stage.calibration.scale_spin.setValue(10.0)       # the template line is 100 px
        stage.flush()
        self.view()
        stage.adjust_video()
        self.assertTrue(stage.editor.set_line((10, 100), (150, 100)))
        stage.flush()
        zones = self.project.entry(self.key).zones
        self.assertAlmostEqual(zones.scale_factor, 10.0 / 140.0)
        stage.set_mode(MODE_TEMPLATE)
        stage.calibration.scale_spin.setValue(20.0)
        stage.flush()
        zones = self.project.entry(self.key).zones
        self.assertAlmostEqual(zones.scale_factor, 20.0 / 140.0)
        self.assertAlmostEqual(
            self.project.effective_settings(self.key).scale_factor, 20.0 / 140.0)
        self.assertAlmostEqual(
            self.project.effective_settings(self.keys[0]).scale_factor, 20.0 / 100.0)


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------
class AlignTests(ArenaFixture):
    def setUp(self):
        super().setUp()
        self.big_keys = self.add_big(2)
        self.stage = self.enter()

    def view(self, key):
        self.stage.set_mode(MODE_VIDEO)
        self.stage.pick_video(key)

    def expected_aligned(self, matrix, dims=BIG):
        return ra.apply_alignment(self.project.template_settings, matrix, dims)

    def test_a_video_of_another_size_asks_to_be_aligned(self):
        self.view(self.big_keys[0])
        stage = self.stage
        self.assertEqual(stage.stack.currentIndex(), PAGE_READONLY)
        kind, text = stage.current_notice()
        self.assertEqual(kind, "warning")
        self.assertIn("200x150", text)
        self.assertIn("160x120", text)
        self.assertEqual(stage.video_chip.text(), "Check alignment")
        ok, reason = stage.can_advance()
        self.assertFalse(ok)
        self.assertIn("2 videos need their zones aligned", reason)

    def test_auto_align_stores_an_aligned_override(self):
        self.view(self.big_keys[0])
        with patch("core.roi_alignment.register_videos",
                   return_value=scaling(1.25)) as register:
            self.assertTrue(self.stage.align_video())
        self.assertEqual(register.call_count, 1)
        template_video, target_video = register.call_args[0][:2]
        self.assertEqual(Path(template_video), self.clips[0])
        self.assertEqual(Path(target_video), self.project.entry(self.big_keys[0]).path)

        entry = self.project.entry(self.big_keys[0])
        self.assertEqual(entry.zones.source, ZONE_ALIGNED)
        self.assertEqual(self.project.zone_status(entry), ZONES_ALIGNED)
        np.testing.assert_allclose(entry.zones.transform, np.diag([1.25, 1.25, 1.0]))
        self.assertEqual(entry.zones.roi_manager.source_dimensions, BIG)
        expected = self.expected_aligned(np.diag([1.25, 1.25, 1.0]))
        self.assertEqual(self.effective_zones(entry.key).to_dict(),
                         expected.roi_manager.to_dict())
        self.assertEqual(self.project.effective_settings(entry.key).scale_factor,
                         expected.scale_factor)
        # The template and the other video are untouched; the badge is "Aligned".
        self.assertIsNone(self.project.entry(self.big_keys[1]).zones)
        badges = {i.key: i.badge_text for i in self.stage.nav.items()}
        self.assertEqual(badges[entry.key], "Aligned")
        self.assertEqual(self.stage.stack.currentIndex(), PAGE_EDITOR)
        self.assertEqual(self.stage.current_notice()[0], "success")

    def test_the_calibration_follows_the_alignment(self):
        state = arena.template_state(self.project)
        state.scale_cm = 25.0
        self.project.set_template(state.to_batch_settings(finalize=True))
        self.app.processEvents()
        self.view(self.big_keys[0])
        with patch("core.roi_alignment.register_videos", return_value=scaling(2.0)):
            self.assertTrue(self.stage.align_video())
        entry = self.project.entry(self.big_keys[0])
        template_scale = self.project.template_settings.scale_factor
        self.assertAlmostEqual(entry.zones.scale_factor, template_scale / 2.0)

    def test_an_uncertain_match_is_flagged_for_a_look(self):
        self.view(self.big_keys[0])
        with patch("core.roi_alignment.register_videos",
                   return_value=scaling(1.25, ra.VERDICT_CHECK)):
            self.assertTrue(self.stage.align_video())
        self.assertEqual(self.stage.current_notice()[0], "warning")

    def test_a_failed_match_changes_nothing_and_says_so(self):
        self.view(self.big_keys[0])
        with patch("core.roi_alignment.register_videos",
                   return_value=failed_registration()):
            self.assertFalse(self.stage.align_video())
        self.assertIsNone(self.project.entry(self.big_keys[0]).zones)
        (title, text), = self.prompts.called("warn")
        self.assertIn("Could not align", title)
        self.assertIn("Adjust zones for this video", text)

    def test_cancelling_changes_nothing(self):
        self.view(self.big_keys[0])
        with patch(RUN, side_effect=TaskCancelled()):
            self.assertFalse(self.stage.align_video())
        self.assertIsNone(self.project.entry(self.big_keys[0]).zones)
        self.assertEqual(self.prompts.called("warn"), [])

    def test_aligning_replaces_adjusted_zones_only_when_confirmed(self):
        key = self.big_keys[0]
        self.view(key)
        self.stage.adjust_video()
        self.prompts.say("confirm", False)
        with patch("core.roi_alignment.register_videos") as register:
            self.assertFalse(self.stage.align_video())
        register.assert_not_called()
        self.assertEqual(self.project.entry(key).zones.source, ZONE_ADJUSTED)
        self.prompts.say("confirm", True)
        with patch("core.roi_alignment.register_videos", return_value=scaling()):
            self.assertTrue(self.stage.align_video())
        self.assertEqual(self.project.entry(key).zones.source, ZONE_ALIGNED)

    def test_reset_removes_an_alignment(self):
        key = self.big_keys[0]
        self.view(key)
        with patch("core.roi_alignment.register_videos", return_value=scaling()):
            self.stage.align_video()
        self.assertTrue(self.stage.reset_video())
        self.assertEqual(self.project.zone_status(key), ZONES_TEMPLATE)

    def test_after_aligning_next_is_allowed(self):
        for key in self.big_keys:
            self.view(key)
            with patch("core.roi_alignment.register_videos", return_value=scaling()):
                self.assertTrue(self.stage.align_video())
        self.assertEqual(self.stage.can_advance(), (True, ""))

    # -- the Tools action ------------------------------------------------------
    def test_the_tools_action_aligns_every_video_that_needs_it(self):
        with patch("core.roi_alignment.register_videos",
                   return_value=scaling(1.25)) as register, \
                patch("gui.workspace.setup_tab.align_template_with_dialog") as dialog:
            self.window.align_action.trigger()
        dialog.assert_not_called()                  # the old handler is gone
        self.assertEqual(register.call_count, 2)
        for key in self.big_keys:
            self.assertEqual(self.project.zone_status(key), ZONES_ALIGNED)
            self.assertEqual(
                self.effective_zones(key).to_dict(),
                self.expected_aligned(np.diag([1.25, 1.25, 1.0])).roi_manager.to_dict())
        for key in self.keys:
            self.assertEqual(self.project.zone_status(key), ZONES_TEMPLATE)
        (title, text), = self.prompts.called("info")
        self.assertIn("Aligned 2 of 2 videos", text)
        self.assertEqual(self.stage.can_advance(), (True, ""))

    def test_the_summary_names_what_failed_and_what_is_uncertain(self):
        results = iter([scaling(1.25, ra.VERDICT_CHECK), failed_registration()])
        with patch("core.roi_alignment.register_videos",
                   side_effect=lambda *a, **k: next(results)):
            self.assertEqual(self.stage.align_all(), 1)
        (title, text), = self.prompts.called("info")
        self.assertIn("Aligned 1 of 2 videos", text)
        self.assertIn("big0.avi", text)
        self.assertIn("uncertain", text)
        self.assertIn("big1.avi", text)
        self.assertIn("not enough of the arena matched", text)
        self.assertEqual(self.project.zone_status(self.big_keys[0]), ZONES_ALIGNED)
        self.assertEqual(self.project.zone_status(self.big_keys[1]), ZONES_TEMPLATE)

    def test_a_crash_on_one_video_does_not_stop_the_others(self):
        calls = []

        def register(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 1:
                raise RuntimeError("cannot decode")
            return scaling()
        with patch("core.roi_alignment.register_videos", side_effect=register),                 self.assertLogs("gui.workspace.stages.arena_stage", "WARNING"):
            self.assertEqual(self.stage.align_all(), 1)
        (title, text), = self.prompts.called("info")
        self.assertIn("cannot decode", text)

    def test_nothing_to_align_says_so(self):
        self.project.remove_videos(self.big_keys)
        self.app.processEvents()
        with patch("core.roi_alignment.register_videos") as register:
            self.assertEqual(self.stage.align_all(), 0)
        register.assert_not_called()
        (title, text), = self.prompts.called("info")
        self.assertIn("fits the template", text)

    def test_adjusted_videos_are_left_alone(self):
        key = self.big_keys[0]
        self.view(key)
        self.stage.adjust_video()
        # An adjusted video of another size is fine; an outdated one is not.
        edited = self.project.template_settings.copy()
        edited.start_time = 0.0
        edited.roi_manager.rois["arm_a"] = [
            ROI(np.array([[2, 2], [40, 2], [40, 40], [2, 40]]), "arm_a")]
        self.project.set_template(edited)       # the template changed afterwards
        self.app.processEvents()
        with patch("core.roi_alignment.register_videos",
                   return_value=scaling()) as register:
            self.stage.align_all()
        self.assertEqual(register.call_count, 1)        # only the other big video
        self.assertEqual(self.project.entry(key).zones.source, ZONE_ADJUSTED)
        (title, text), = self.prompts.called("info")
        self.assertIn("big0.avi", text)
        self.assertIn("by hand", text)

    def test_without_a_template_the_action_explains(self):
        self.project.set_template(None)
        self.app.processEvents()
        self.assertFalse(self.window.align_action.isEnabled())
        self.assertEqual(self.stage.align_all(), 0)
        (title, text), = self.prompts.called("info")
        self.assertIn("template", text)

    def test_the_action_follows_the_template(self):
        self.assertTrue(self.window.align_action.isEnabled())
        self.project.set_template(None)
        self.app.processEvents()
        self.assertFalse(self.window.align_action.isEnabled())
        self.project.set_template(valid_settings(Y))
        self.app.processEvents()
        self.assertTrue(self.window.align_action.isEnabled())

    def test_the_template_video_is_asked_for_when_none_fits(self):
        self.project.remove_videos(self.keys)
        self.app.processEvents()
        self.prompts.say("open_file", self.clips[0])
        with patch("core.roi_alignment.register_videos", return_value=scaling()):
            self.assertEqual(self.stage.align_all(), 2)
        (title, _start), = self.prompts.called("open_file")
        self.assertIn("template", title)

    def test_a_wrongly_sized_template_video_is_refused(self):
        self.project.remove_videos(self.keys)
        self.app.processEvents()
        self.prompts.say("open_file", self.project.entry(self.big_keys[0]).path)
        with patch("core.roi_alignment.register_videos") as register:
            self.assertEqual(self.stage.align_all(), 0)
        register.assert_not_called()
        self.assertEqual(len(self.prompts.called("warn")), 1)


class RealAlignmentTests(ArenaFixture):
    """The whole path with real frames and the real registration."""

    TEMPLATE = False

    def write(self, name, image):
        path = self.root / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (image.shape[1], image.shape[0]))
        for _ in range(12):
            writer.write(image)
        writer.release()
        return path

    def test_a_moved_camera_is_aligned_from_the_pictures(self):
        scene = arena_image()
        truth = camera_move(shift=(14.0, -9.0), rotation_deg=3.0, scale=1.04)
        reference = self.write("ref.avi", scene)
        moved = self.write("moved.avi", warp(scene, truth, (640, 480)))
        self.project.remove_videos(self.keys)
        self.project.add_videos([reference, moved])
        self.project.set_template(settings_for(Y, size=(640, 480)))
        self.app.processEvents()
        stage = self.enter()
        key = self.project.keys()[1]
        stage.set_mode(MODE_VIDEO)
        stage.pick_video(key)
        threads = threading.active_count()
        self.assertTrue(stage.align_video())
        self.assertEqual(threading.active_count(), threads)
        entry = self.project.entry(key)
        self.assertEqual(entry.zones.source, ZONE_ALIGNED)
        self.assertLess(max_grid_error(np.array(entry.zones.transform), truth), 4.0)
        template = self.project.template_settings.roi_manager
        moved_zones = self.effective_zones(key)
        expected = ra.transform_points(
            truth, template.rois["arm_a"][0].points.astype(float))
        np.testing.assert_allclose(
            moved_zones.rois["arm_a"][0].points, expected, atol=5.0)


# ----------------------------------------------------------------------
# Footer, status and threads
# ----------------------------------------------------------------------
class FooterTests(ArenaFixture):
    def test_the_footer_counts_template_adjusted_and_check(self):
        big = self.add_big(1)
        stage = self.enter()
        stage.set_mode(MODE_VIDEO)
        stage.pick_video(self.keys[1])
        stage.adjust_video()
        text, chip = stage.footer_text()
        self.assertIn("2 videos use the template", text)
        self.assertIn("1 video with its own zones", text)
        self.assertIn("1 video to check", text)
        self.assertEqual(chip, ("warning", "Check alignment"))
        self.assertEqual(self.window.footer.next_text(), "Next: Detection →")
        self.assertFalse(self.window.footer.next_button.isEnabled())
        with patch("core.roi_alignment.register_videos", return_value=scaling()):
            stage.pick_video(big[0])
            stage.align_video()
        text, chip = stage.footer_text()
        self.assertIn("1 video uses the template", text)
        self.assertIn("2 videos with their own zones", text)
        self.assertNotIn("to check", text)
        self.assertEqual(chip, ("ok", "Zones set"))
        self.assertTrue(self.window.footer.next_button.isEnabled())

    def test_a_happy_study(self):
        stage = self.enter()
        text, chip = stage.footer_text()
        self.assertEqual(text, "2 videos use the template")
        self.assertEqual(stage.status()[0], "done")
        self.assertEqual(stage.can_advance(), (True, ""))

    def test_the_stepper_follows(self):
        big = self.add_big(1)
        stage = self.enter()
        self.assertEqual(self.window.stepper.status("arena"), "current")
        self.assertEqual(stage.status()[0], "attention")
        with patch("core.roi_alignment.register_videos", return_value=scaling()):
            stage.set_mode(MODE_VIDEO)
            stage.pick_video(big[0])
            stage.align_video()
        self.assertEqual(stage.status()[0], "done")

    def test_next_is_blocked_without_a_template(self):
        self.project.set_template(None)
        self.app.processEvents()
        stage = self.enter()
        ok, reason = stage.can_advance()
        self.assertFalse(ok)
        self.assertEqual(stage.status()[0], "todo")
        self.assertFalse(self.window.footer.next_button.isEnabled())
        self.assertTrue(self.window.footer.next_button.toolTip())


class LifecycleTests(ArenaFixture):
    def test_no_thread_outlives_a_visit(self):
        before = threading.active_count()
        big = self.add_big(1)
        stage = self.enter()
        stage.set_mode(MODE_VIDEO)
        stage.pick_video(big[0])
        with patch("core.roi_alignment.register_videos", return_value=scaling()):
            stage.align_video()
            self.window.align_action.trigger()
        self.window.thumbnails.wait_idle()
        self.window.go_to_stage("videos")
        self.app.processEvents()
        # The thumbnail loader keeps one pool thread alive until the window
        # closes; nothing of the stage's own may remain.
        self.assertLessEqual(threading.active_count(), before + 1)

    def test_the_timer_belongs_to_the_stage(self):
        self.assertIs(self.stage._timer.parent(), self.stage)

    def test_a_new_project_drops_pending_edits(self):
        stage = self.enter()
        stage.editor.set_current_zone("arm_a")
        stage.editor.clear_current_zone()
        draw_polygon(stage.editor, "arm_a", [(2, 60), (50, 60), (50, 110), (2, 110)])
        self.assertTrue(stage.has_pending_writes())
        self.prompts.say("ask_save_changes", "discard")
        self.assertTrue(self.window.close_project())
        self.assertFalse(stage.has_pending_writes())
        self.assertIsNone(stage.editor)

    def test_changing_the_paradigm_rebuilds_the_stage(self):
        stage = self.enter()
        self.assertIsNotNone(stage.editor)
        self.prompts.say("confirm", True)
        self.window.change_paradigm(E)
        self.app.processEvents()
        self.assertIsNone(self.project.template_settings)
        self.assertEqual(stage.state.maze_name, E)
        self.assertEqual(
            [r.category for r in stage.editor.session.rows()],
            arm_zones(create_maze(E)))

    def test_removing_the_picked_video_picks_another(self):
        stage = self.enter()
        stage.pick_video(self.keys[1])
        self.project.remove_videos([self.keys[1]])
        self.app.processEvents()
        self.assertEqual(stage.selected_key, self.keys[0])

    def test_an_unreadable_video_gets_a_message_not_a_crash(self):
        project = self.project
        bad = self.write_video("videos/bad.avi", b"not a video")
        project.add_videos([bad])
        self.app.processEvents()
        stage = self.enter()
        stage.pick_video(project.keys()[-1])
        self.assertEqual(stage.stack.currentIndex(), PAGE_MESSAGE)
        self.assertIn("cannot be shown", stage.message.title())
        self.assertEqual(len(stage.message.buttons), 2)
        stage.pick_video(self.keys[0])
        self.assertEqual(stage.stack.currentIndex(), PAGE_EDITOR)

    def test_the_detection_stage_is_its_own_stage(self):
        # The old setup tab is no longer hosted: Detection is a real stage.
        from gui.workspace.stages.detection_stage import DetectionStage
        self.window.go_to_stage("detection")
        self.assertIsInstance(self.window.stage("detection"), DetectionStage)
        self.assertIsNot(self.window.setup_tab.parentWidget(),
                         self.window.stage("detection").content)


if __name__ == "__main__":
    unittest.main()
